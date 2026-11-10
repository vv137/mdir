//===- CompileCache.cpp - The compile cache (D212) ----------------------===//
//
// The object cache follows the interface that LLVM's ORC JIT offers for it
// (llvm::ObjectCache, as in LLVM's LLJITWithObjectCache example); entries
// are content-addressed by the module they were generated from.
//
//===----------------------------------------------------------------------===//

#include "mdir/Compiler/CompileCache.h"
#include "mdir/Runtime/GroupsABI.h"
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
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace llvm;
using namespace mdir::compiler;

namespace {
// The layout of an entry; a change of it changes this tag, so old entries
// are rejected (and replaced) rather than misread.
constexpr char kMagic[8] = {'M', 'D', 'I', 'R', 'O', 'B', 'J', '1'};
constexpr StringLiteral kFormat = "mdir-object-cache 2";
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
// How long a store waits for the lock of `size` before it lists the
// directory without it: a holder adds to the total or lists the directory,
// which takes seconds on a network file system at most.
constexpr auto kLockTimeout = std::chrono::milliseconds(10000);
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

namespace {
/// Writes `message` to the standard error once per process.
void warnOnce(const std::string &message) {
  static std::mutex mutex;
  static std::map<std::string, bool> said;
  std::lock_guard<std::mutex> lock(mutex);
  if (said.emplace(message, true).second)
    errs() << "mdir: " << message << "\n";
}

std::string getEnvironment(const char *name) {
  auto value = sys::Process::GetEnv(name);
  return value ? *value : std::string();
}

std::string describeErrno(int error) {
  return std::error_code(error, std::generic_category()).message();
}

Error errnoToError(int error) {
  return errorCodeToError(std::error_code(error, std::generic_category()));
}

/// A file descriptor, closed with its owner.
class Descriptor {
public:
  explicit Descriptor(int fd = -1) : fd(fd) {}
  ~Descriptor() {
    if (fd >= 0)
      ::close(fd);
  }
  Descriptor(Descriptor &&other) : fd(other.release()) {}
  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;
  int get() const { return fd; }
  int release() { return std::exchange(fd, -1); }

private:
  int fd;
};

/// Opens the directory `name` relative to `at`, without following a link.
int openDirectoryAt(int at, const char *name) {
  return ::openat(at, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
}

/// Whether the file system of `path`, or of the nearest directory above it
/// that exists, is local. MDIR_COMPILE_CACHE_TEST_LOCAL=1, for tests only,
/// takes every one as local.
bool isLocal(StringRef path) {
  if (getEnvironment("MDIR_COMPILE_CACHE_TEST_LOCAL") == "1")
    return true;
  SmallString<256> existing(path);
  while (!existing.empty() && !sys::fs::exists(existing))
    sys::path::remove_filename(existing);
  bool local = false;
  return !existing.empty() && !sys::fs::is_local(existing, local) && local;
}

/// The default directory (D397):
/// `${XDG_CACHE_HOME:-$HOME/.cache}/mdir/compile` if its file system is
/// local, else `/var/tmp/mdir-<uid>/compile` if local; nullopt with the
/// reason otherwise.
std::optional<std::string> findDefaultDirectory(std::string &why) {
  std::vector<std::string> bases;
  // A relative XDG_CACHE_HOME is ignored, as the XDG specification says.
  std::string xdg = getEnvironment("XDG_CACHE_HOME");
  std::string home = getEnvironment("HOME");
  if (sys::path::is_absolute(xdg)) {
    bases.push_back(xdg);
  } else if (sys::path::is_absolute(home)) {
    SmallString<256> path(home);
    sys::path::append(path, ".cache");
    bases.push_back(std::string(path));
  }
  std::string user = "mdir-" + std::to_string(::geteuid());
  for (StringRef base : bases) {
    SmallString<256> path(base);
    sys::path::append(path, "mdir", "compile");
    if (isLocal(path))
      return std::string(path);
    why += "'" + base.str() + "' is not on a local file system; ";
  }
  SmallString<256> path("/var/tmp");
  sys::path::append(path, user, "compile");
  if (isLocal(path))
    return std::string(path);
  why += "'/var/tmp' is not on a local file system; ";
  return std::nullopt;
}

/// Opens the component `name` of a default directory relative to `at`
/// (`shown` is its path, for messages), made with mode 0700 first if
/// `create`. Refused, -1 with the reason in `refused`, if it is a symbolic
/// link or not a directory, or another user owns it; if group or others may
/// write it, it loses that permission when `tighten`, and is refused
/// otherwise. Whoever may write a parent could put a directory of their own
/// in the place of the one below it.
int openPrivateComponent(int at, StringRef name, const std::string &shown,
                         bool create, bool tighten, std::string &refused) {
  SmallString<64> component(name);
  if (create && ::mkdirat(at, component.c_str(), 0700) != 0 &&
      errno != EEXIST) {
    refused = "cannot make '" + shown + "': " + describeErrno(errno);
    return -1;
  }
  Descriptor fd(openDirectoryAt(at, component.c_str()));
  if (fd.get() < 0) {
    int error = errno;
    struct stat status;
    if (::fstatat(at, component.c_str(), &status, AT_SYMLINK_NOFOLLOW) == 0 &&
        S_ISLNK(status.st_mode))
      refused = "'" + shown + "' is a symbolic link";
    else if (error == ENOTDIR)
      refused = "'" + shown + "' is not a directory";
    else if (error == ENOENT)
      refused = "'" + shown + "' does not exist";
    else
      refused = "'" + shown + "' cannot be opened: " + describeErrno(error);
    return -1;
  }
  struct stat status;
  if (::fstat(fd.get(), &status) != 0) {
    refused = "'" + shown + "' cannot be read";
    return -1;
  }
  if (status.st_uid != ::geteuid()) {
    refused = "'" + shown + "' belongs to user " +
              std::to_string(status.st_uid);
    return -1;
  }
  if (status.st_mode & (S_IWGRP | S_IWOTH)) {
    if (!tighten || ::fchmod(fd.get(), status.st_mode & 07777 &
                                           ~(S_IWGRP | S_IWOTH)) != 0) {
      refused = "group or others may write '" + shown + "'";
      return -1;
    }
  }
  return fd.release();
}
} // namespace

CacheDirectory::CacheDirectory(std::string path) : path(std::move(path)) {}

CacheDirectory::~CacheDirectory() {
  if (descriptor >= 0)
    ::close(descriptor);
  for (const auto &entry : subdirectories)
    ::close(entry.second);
}

std::shared_ptr<CacheDirectory>
CacheDirectory::openDefault(StringRef path, bool create,
                            std::string &refused) {
  SmallString<256> parent(sys::path::parent_path(path));
  SmallString<256> base(sys::path::parent_path(parent));
  StringRef leaf = sys::path::filename(path);
  StringRef parentName = sys::path::filename(parent);
  // The base is the user's ($XDG_CACHE_HOME, $HOME/.cache) or the system's
  // (/var/tmp), and is taken as given, links included.
  if (create) {
    if (std::error_code error = sys::fs::create_directories(
            base, /*IgnoreExisting=*/true, sys::fs::perms::owner_all)) {
      refused = "cannot make '" + base.str().str() + "': " + error.message();
      return nullptr;
    }
  }
  Descriptor baseFd(
      ::open(base.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (baseFd.get() < 0) {
    refused = errno == ENOENT ? "'" + parent.str().str() + "' does not exist"
                              : "'" + base.str().str() +
                                    "' cannot be opened: " +
                                    describeErrno(errno);
    return nullptr;
  }
  Descriptor parentFd(openPrivateComponent(baseFd.get(), parentName,
                                           parent.str().str(), create,
                                           /*tighten=*/true, refused));
  if (parentFd.get() < 0)
    return nullptr;
  int fd = openPrivateComponent(parentFd.get(), leaf, path.str(), create,
                                /*tighten=*/false, refused);
  if (fd < 0)
    return nullptr;
  std::shared_ptr<CacheDirectory> directory(new CacheDirectory());
  directory->path = path.str();
  directory->isDefault = true;
  directory->descriptor = fd;
  return directory;
}

int CacheDirectory::getDescriptor(bool create) {
  std::lock_guard<std::mutex> lock(mutex);
  if (descriptor >= 0 || isDefault)
    return descriptor;
  // A named directory: as given, made as `create_directories` makes it.
  if (create && sys::fs::create_directories(path))
    return -1;
  descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  return descriptor;
}

int CacheDirectory::getSubdirectory(StringRef kind, bool create) {
  int root = getDescriptor(create);
  if (root < 0)
    return -1;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto found = subdirectories.find(kind.str());
      found != subdirectories.end())
    return found->second;
  SmallString<16> name(kind);
  // Private in the default directory; as the umask says in a named one.
  if (create && ::mkdirat(root, name.c_str(), isDefault ? 0700 : 0777) != 0 &&
      errno != EEXIST)
    return -1;
  int fd = openDirectoryAt(root, name.c_str());
  if (fd >= 0)
    subdirectories[kind.str()] = fd;
  return fd;
}

namespace {
CompileCacheLocation computeLocation() {
  CompileCacheLocation location;
  if (StringRef(getEnvironment("MDIR_COMPILE_CACHE")).equals_insensitive("off")) {
    location.reason = "MDIR_COMPILE_CACHE=off";
    return location;
  }
  CompileCacheConfig config;
  if (auto size = sys::Process::GetEnv("MDIR_COMPILE_CACHE_MAX_MB")) {
    uint64_t megabytes = 0;
    // A bound whose bytes do not fit in 64 bits is no bound.
    if (StringRef(*size).trim().getAsInteger(10, megabytes) ||
        megabytes > (std::numeric_limits<uint64_t>::max() >> 20)) {
      warnOnce("warning: MDIR_COMPILE_CACHE_MAX_MB='" + *size +
               "' is not a whole number of MiB; the compile cache keeps its "
               "bound of 2048 MiB");
    } else if (megabytes == 0) {
      location.reason = "MDIR_COMPILE_CACHE_MAX_MB=0";
      return location;
    } else {
      config.maxBytes = megabytes << 20;
    }
  }
  // The salt holds the value of every LLVM option set on the command line;
  // one whose value it could not record would let entries of two values
  // meet (D397).
  if (std::vector<std::string> unrecorded = getUnrecordedOptions();
      !unrecorded.empty()) {
    location.reason = "the keys do not record the value of the LLVM option";
    if (unrecorded.size() > 1)
      location.reason += "s";
    for (size_t i = 0; i < unrecorded.size(); ++i)
      location.reason += (i ? ", -" : " -") + unrecorded[i];
    warnOnce("note: the compile cache is off: " + location.reason);
    return location;
  }
  std::string named = getEnvironment("MDIR_COMPILE_CACHE_DIR");
  if (!named.empty()) {
    config.directory = named;
    config.opened = std::make_shared<CacheDirectory>(named);
    struct stat status;
    if (::stat(named.c_str(), &status) == 0 &&
        (status.st_mode & (S_IWGRP | S_IWOTH)))
      warnOnce("warning: group or others may write the compile cache '" +
               named + "' (MDIR_COMPILE_CACHE_DIR); its host objects are "
               "code that runs, so anyone who may write it can run code as "
               "you");
    location.config = std::move(config);
    location.source = "MDIR_COMPILE_CACHE_DIR";
    return location;
  }
  std::string why;
  auto directory = findDefaultDirectory(why);
  if (!directory) {
    why.resize(why.size() - 2);
    location.reason = "no default directory: " + why +
                      "; MDIR_COMPILE_CACHE_DIR names one";
    warnOnce("note: the compile cache is off: " + location.reason);
    return location;
  }
  // Made private (0700), opened once, and checked through its descriptor
  // before anything is read or written; every entry is then opened
  // relative to that descriptor.
  std::string refused;
  config.opened = CacheDirectory::openDefault(*directory, /*create=*/true,
                                              refused);
  if (!config.opened) {
    location.reason = "refused the default directory: " + refused;
    warnOnce("warning: the compile cache is off: " + location.reason +
             "; nothing is read from it or written to it");
    return location;
  }
  config.directory = *directory;
  location.config = std::move(config);
  location.source = "default";
  return location;
}

/// The open directory of `config`, opened as named if it has none.
CacheDirectory &getOpened(CompileCacheConfig &config) {
  if (!config.opened)
    config.opened = std::make_shared<CacheDirectory>(config.directory);
  return *config.opened;
}
} // namespace

CompileCacheLocation CompileCacheConfig::locate() {
  // Once per process for each environment that decides it.
  std::string environment;
  for (const char *name :
       {"MDIR_COMPILE_CACHE", "MDIR_COMPILE_CACHE_DIR",
        "MDIR_COMPILE_CACHE_MAX_MB", "XDG_CACHE_HOME", "HOME",
        "MDIR_COMPILE_CACHE_TEST_LOCAL"}) {
    auto value = sys::Process::GetEnv(name);
    environment += value ? "=" + *value : std::string("-");
    environment += '\0';
  }
  static std::mutex mutex;
  static std::map<std::string, CompileCacheLocation> located;
  std::lock_guard<std::mutex> lock(mutex);
  auto found = located.find(environment);
  if (found == located.end())
    found = located.emplace(environment, computeLocation()).first;
  return found->second;
}

std::optional<CompileCacheConfig> CompileCacheConfig::fromEnvironment() {
  return locate().config;
}

std::shared_ptr<CacheDirectory>
mdir::compiler::openCacheDirectoryToClear(std::string &path) {
  path = getEnvironment("MDIR_COMPILE_CACHE_DIR");
  if (!path.empty())
    return std::make_shared<CacheDirectory>(path);
  std::string why;
  auto directory = findDefaultDirectory(why);
  if (!directory)
    return nullptr;
  path = *directory;
  std::string refused;
  auto opened = CacheDirectory::openDefault(path, /*create=*/false, refused);
  // A default that does not exist yet holds nothing to clear.
  if (!opened && !StringRef(refused).ends_with("' does not exist"))
    warnOnce("warning: the compile cache is not cleared: refused the "
             "default directory: " + refused);
  return opened;
}

std::string mdir::compiler::describeMachine(const TargetMachine &machine,
                                            StringRef extra) {
  std::string text;
  raw_string_ostream os(text);
  const TargetOptions &o = machine.Options;
  os << kFormat << '\n'
     << getCacheSalt()
     << "groups-abi " << MDRT_GROUPS_ABI_VERSION << '\n'
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
      module(std::move(module)) {
  if (this->config)
    getOpened(*this->config);
}

HostObjectCache::~HostObjectCache() = default;

std::string HostObjectCache::getKey(const Module &module) const {
  HashStream stream;
  WriteBitcodeToFile(module, stream, /*ShouldPreserveUseListOrder=*/true);
  // The scheduler in force now, as the code generator that follows will
  // find it.
  return machine + "sched " + getSchedulerInForce() + "\nmodule " +
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

namespace {
/// Reads the entry `name` relative to the directory `at`, as
/// readCacheEntry does.
std::unique_ptr<MemoryBuffer>
readEntryAt(int at, StringRef name, StringRef key, double &seconds,
            bool &rejected, function_ref<bool(StringRef)> isValid) {
  rejected = false;
  if (at < 0)
    return nullptr;
  SmallString<128> component(name);
  Descriptor fd(::openat(at, component.c_str(),
                         O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
  if (fd.get() < 0) {
    // A link in the place of an entry is no entry of this cache.
    rejected = errno == ELOOP;
    return nullptr;
  }
  rejected = true;
  struct stat status;
  if (::fstat(fd.get(), &status) != 0 || !S_ISREG(status.st_mode))
    return nullptr;
  // Read, not mapped: another process may replace the file by a rename.
  auto file = MemoryBuffer::getOpenFile(fd.get(), name, status.st_size,
                                        /*RequiresNullTerminator=*/false,
                                        /*IsVolatile=*/true);
  if (!file)
    return nullptr;
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
  auto data = MemoryBuffer::getMemBufferCopy(in, name);
  if (isValid && !isValid(data->getBuffer()))
    return nullptr;
  rejected = false;
  return data;
}

/// Writes the entry `name` relative to the directory `at`, as
/// writeCacheEntry does.
Error writeEntryAt(int at, StringRef name, StringRef key, StringRef object,
                   double seconds, uint64_t *written) {
  if (at < 0)
    return errnoToError(ENOENT);
  std::string data(kMagic, sizeof(kMagic));
  append<uint64_t>(data, key.size());
  data += key;
  append<uint64_t>(data, object.size());
  append<double>(data, seconds);
  data += hashObject(object);
  data += object;
  // A unique temporary file in the same directory, then a rename: a
  // reader sees the old entry, the new one, or none, never a part.
  std::string temporary;
  int fd = -1;
  for (int attempt = 0; attempt < 64 && fd < 0; ++attempt) {
    temporary = name.str() + ".tmp-";
    for (int i = 0; i < 3; ++i)
      temporary += utohexstr(sys::Process::GetRandomNumber() & 0xffff,
                             /*LowerCase=*/true, /*Width=*/4);
    fd = ::openat(at, temporary.c_str(),
                  O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0666);
    if (fd < 0 && errno != EEXIST)
      return errnoToError(errno);
  }
  if (fd < 0)
    return errnoToError(EEXIST);
  Descriptor file(fd);
  for (size_t done = 0; done < data.size();) {
    ssize_t count = ::write(fd, data.data() + done, data.size() - done);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0) {
      int error = count < 0 ? errno : EIO;
      ::unlinkat(at, temporary.c_str(), 0);
      return errnoToError(error);
    }
    done += count;
  }
  if (::close(file.release()) != 0) {
    int error = errno;
    ::unlinkat(at, temporary.c_str(), 0);
    return errnoToError(error);
  }
  SmallString<128> target(name);
  if (::renameat(at, temporary.c_str(), at, target.c_str()) != 0) {
    int error = errno;
    ::unlinkat(at, temporary.c_str(), 0);
    return errnoToError(error);
  }
  if (written)
    *written = data.size();
  return Error::success();
}

/// The directory of `path` opened as given, and the name of its file.
std::pair<Descriptor, std::string> openParentOf(StringRef path) {
  SmallString<256> parent(sys::path::parent_path(path));
  if (parent.empty())
    parent = ".";
  return {Descriptor(::open(parent.c_str(), O_RDONLY | O_DIRECTORY |
                                                O_CLOEXEC)),
          sys::path::filename(path).str()};
}

bool isHostObject(StringRef data) {
  // An object LLVM cannot read is rejected here rather than at linking.
  auto parsed = object::ObjectFile::createObjectFile(
      MemoryBufferRef(data, "cached object"));
  if (!parsed) {
    consumeError(parsed.takeError());
    return false;
  }
  return true;
}
} // namespace

std::unique_ptr<MemoryBuffer> mdir::compiler::readCacheEntry(
    CacheDirectory &directory, StringRef kind, StringRef name, StringRef key,
    double &seconds, bool &rejected, function_ref<bool(StringRef)> isValid) {
  return readEntryAt(directory.getSubdirectory(kind, /*create=*/false), name,
                     key, seconds, rejected, isValid);
}

std::unique_ptr<MemoryBuffer>
mdir::compiler::readCacheEntry(StringRef path, StringRef key, double &seconds,
                               bool &rejected,
                               function_ref<bool(StringRef)> isValid) {
  auto [parent, name] = openParentOf(path);
  return readEntryAt(parent.get(), name, key, seconds, rejected, isValid);
}

std::unique_ptr<MemoryBuffer>
HostObjectCache::readEntry(StringRef path, StringRef key, double &seconds,
                           bool &rejected) {
  return readCacheEntry(path, key, seconds, rejected, isHostObject);
}

Error HostObjectCache::writeEntry(StringRef path, StringRef key,
                                  MemoryBufferRef object, double seconds,
                                  uint64_t *written) {
  return writeCacheEntry(path, key, object.getBuffer(), seconds, written);
}

Error mdir::compiler::writeCacheEntry(CacheDirectory &directory,
                                      StringRef kind, StringRef name,
                                      StringRef key, StringRef object,
                                      double seconds, uint64_t *written) {
  return writeEntryAt(directory.getSubdirectory(kind, /*create=*/true), name,
                      key, object, seconds, written);
}

Error mdir::compiler::writeCacheEntry(StringRef path, StringRef key,
                                      StringRef object, double seconds,
                                      uint64_t *written) {
  auto [parent, name] = openParentOf(path);
  return writeEntryAt(parent.get(), name, key, object, seconds, written);
}

std::string mdir::compiler::getCacheEntryName(StringRef key,
                                              StringRef extension) {
  return hashObject(key) + extension.str();
}

void mdir::compiler::touchCacheEntry(CacheDirectory &directory, StringRef kind,
                                     StringRef name) {
  int at = directory.getSubdirectory(kind, /*create=*/false);
  SmallString<128> component(name);
  if (at >= 0)
    (void)::utimensat(at, component.c_str(), nullptr, AT_SYMLINK_NOFOLLOW);
}

namespace {
/// Calls `visit(name, status)` for each regular file of the directory `at`,
/// listed through a descriptor of its own; links are not followed.
template <typename Visit> void forEachFile(int at, Visit visit) {
  if (at < 0)
    return;
  int fd = ::openat(at, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0)
    return;
  DIR *listing = ::fdopendir(fd);
  if (!listing) {
    ::close(fd);
    return;
  }
  while (struct dirent *entry = ::readdir(listing)) {
    struct stat status;
    // Another process may have removed it since the listing.
    if (::fstatat(at, entry->d_name, &status, AT_SYMLINK_NOFOLLOW) != 0 ||
        !S_ISREG(status.st_mode))
      continue;
    visit(StringRef(entry->d_name), status);
  }
  ::closedir(listing);
}

sys::TimePoint<> getModificationTime(const struct stat &status) {
  return sys::toTimePoint(status.st_mtim.tv_sec, status.st_mtim.tv_nsec);
}

/// Lists the entries of every kind, removes the temporary files left by
/// processes that died and, when they exceed `maxBytes`, entries least
/// recently used first until at most `target` (at most `maxBytes`) are
/// left. Returns the bytes left.
uint64_t listAndEvict(CacheDirectory &directory, uint64_t maxBytes,
                      uint64_t target) {
  ++listings;
  struct Entry {
    int at;
    std::string name;
    uint64_t size;
    sys::TimePoint<> used;
  };
  std::vector<Entry> entries;
  uint64_t total = 0;
  auto current = std::chrono::system_clock::now();
  // One bound for the entries of every kind.
  for (StringRef kind : {"host", "gpu"}) {
    int at = directory.getSubdirectory(kind, /*create=*/false);
    forEachFile(at, [&](StringRef name, const struct stat &status) {
      if (name.contains(".tmp-")) {
        if (current - getModificationTime(status) > kStaleTemporary)
          ::unlinkat(at, name.data(), 0);
        return;
      }
      if (!name.ends_with(".o") && !name.ends_with(".ptx") &&
          !name.ends_with(".cubin"))
        return;
      entries.push_back({at, name.str(), uint64_t(status.st_size),
                         getModificationTime(status)});
      total += status.st_size;
    });
  }
  if (total <= maxBytes)
    return total;
  std::sort(entries.begin(), entries.end(),
            [](const Entry &a, const Entry &b) { return a.used < b.used; });
  for (const Entry &entry : entries) {
    if (total <= target)
      break;
    // A file another process removed first counts as removed.
    ::unlinkat(entry.at, entry.name.c_str(), 0);
    total -= entry.size;
  }
  return total;
}

/// Locks the file `fd` for writing between processes (fcntl), waiting at
/// most `timeout`, with pauses that double up to 64 ms: on a network file
/// system each try is a round trip to the server. Returns whether it holds
/// the lock.
bool lockWithin(int fd, std::chrono::milliseconds timeout) {
  auto end = std::chrono::steady_clock::now() + timeout;
  auto pause = std::chrono::milliseconds(1);
  while (true) {
    struct flock lock;
    std::memset(&lock, 0, sizeof(lock));
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    if (::fcntl(fd, F_SETLK, &lock) != -1)
      return true;
    if (errno != EACCES && errno != EAGAIN)
      return false;
    auto current = std::chrono::steady_clock::now();
    if (current >= end)
      return false;
    std::this_thread::sleep_for(
        std::min<std::chrono::steady_clock::duration>(pause, end - current));
    pause = std::min(pause * 2, std::chrono::milliseconds(64));
  }
}

/// The file of the total of the cache at `directory`, open and locked
/// against other processes and other threads until it is destroyed. A
/// process holds it only while it adds to the total or lists the directory.
class SizeFile {
public:
  SizeFile(CacheDirectory &directory, bool create) : lock(getMutex()) {
    int at = directory.getDescriptor(/*create=*/false);
    if (at < 0)
      return;
    fd = ::openat(at, "size",
                  O_RDWR | O_NOFOLLOW | O_CLOEXEC | (create ? O_CREAT : 0),
                  0666);
    if (fd < 0)
      return;
    // A lock of the file between processes (fcntl), which a network file
    // system carries; a file system without locks, or a lock not given
    // within the timeout, leaves the file unused: the store then lists the
    // directory, as one without locks does (D397).
    if (!lockWithin(fd, kLockTimeout)) {
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

/// Where the eviction of a store at the bound stops: 90% of the bound, so
/// that the stores after it fit without a listing until a tenth of the
/// bound is written again (#675). Evicting only to the bound left the
/// cache at it, and every later store listed the directory while it held
/// the lock of `size`, which serialized a suite on a network file system.
uint64_t lowWater(uint64_t maxBytes) { return maxBytes - maxBytes / 10; }

int64_t secondsSinceEpoch() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
} // namespace

void mdir::compiler::evictCache(CacheDirectory &directory, uint64_t maxBytes) {
  SizeFile file(directory, /*create=*/false);
  uint64_t total = listAndEvict(directory, maxBytes, maxBytes);
  if (file.usable())
    file.write(total, secondsSinceEpoch());
}

void mdir::compiler::evictCache(StringRef directory, uint64_t maxBytes) {
  CacheDirectory opened(directory.str());
  evictCache(opened, maxBytes);
}

void mdir::compiler::noteCacheStores(CacheDirectory &directory, uint64_t bytes,
                                     uint64_t maxBytes) {
  SizeFile file(directory, /*create=*/true);
  if (!file.usable()) {
    // No total to rely on: the directory is listed, as after every store
    // before the total was kept.
    listAndEvict(directory, maxBytes, lowWater(maxBytes));
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
  file.write(listAndEvict(directory, maxBytes, lowWater(maxBytes)), current);
}

void mdir::compiler::noteCacheStores(StringRef directory, uint64_t bytes,
                                     uint64_t maxBytes) {
  CacheDirectory opened(directory.str());
  noteCacheStores(opened, bytes, maxBytes);
}

uint64_t mdir::compiler::getCacheListingCount() { return listings; }

ClearedCache mdir::compiler::clearCache(CacheDirectory &directory) {
  ClearedCache cleared;
  auto current = std::chrono::system_clock::now();
  for (StringRef kind : {"host", "gpu"}) {
    int at = directory.getSubdirectory(kind, /*create=*/false);
    forEachFile(at, [&](StringRef name, const struct stat &status) {
      // The temporary file of a writer under way is left, so that its
      // rename succeeds; one older than an hour was left by a process
      // that died.
      if (name.contains(".tmp-")) {
        if (current - getModificationTime(status) > kStaleTemporary)
          ::unlinkat(at, name.data(), 0);
        return;
      }
      if (!name.ends_with(".o") && !name.ends_with(".ptx") &&
          !name.ends_with(".cubin"))
        return;
      // Only entries of this format: the magic tag of the layout.
      Descriptor fd(::openat(at, name.data(), O_RDONLY | O_NOFOLLOW |
                                                  O_CLOEXEC));
      char head[sizeof(kMagic)];
      if (fd.get() < 0 ||
          ::pread(fd.get(), head, sizeof(head), 0) !=
              static_cast<ssize_t>(sizeof(head)) ||
          std::memcmp(head, kMagic, sizeof(kMagic)) != 0)
        return;
      // A file another process removed first counts as left to it.
      if (::unlinkat(at, name.data(), 0) != 0)
        return;
      ++(kind == "host" ? cleared.hostEntries : cleared.gpuEntries);
      cleared.bytes += status.st_size;
    });
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

ClearedCache mdir::compiler::clearCache(StringRef directory) {
  CacheDirectory opened(directory.str());
  return clearCache(opened);
}

std::unique_ptr<MemoryBuffer> HostObjectCache::getObject(const Module *module) {
  if (module->getModuleIdentifier() != this->module)
    return nullptr;
  // The code that the program keeps: a copy of its object, which the
  // engine links into memory of its own (D236).
  if (kept)
    return MemoryBuffer::getMemBufferCopy(kept->object, this->module);
  // The key is the bitcode of the module before code generation, which
  // changes the module; without a directory, only the generation is timed.
  Pending request;
  double start = now();
  if (config) {
    request.key = getKey(*module);
    std::string name = getEntryName(request.key);
    double seconds = 0.0;
    bool rejected = false;
    if (auto object = readCacheEntry(*config->opened, "host", name,
                                     request.key, seconds, rejected,
                                     isHostObject)) {
      // Marks the entry as used, for the eviction of the least recent.
      touchCacheEntry(*config->opened, "host", name);
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
    // A cache that cannot be written is no cache; the run goes on.
    if (auto error = writeCacheEntry(*config->opened, "host",
                                     getEntryName(request.key), request.key,
                                     object.getBuffer(), seconds, &written)) {
      consumeError(std::move(error));
    } else {
      stored = true;
      noteCacheStores(*config->opened, written, config->maxBytes);
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
