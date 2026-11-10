//===- CompileCacheSalt.cpp - What every key of the cache begins with -----===//
//
// The salt that names the build, the options of LLVM, and the scheduler in
// every key of the compile cache (D[compile-cache-default]), and
// the hashes of the files of the toolchain that the keys hold.
//
//===----------------------------------------------------------------------===//

#include "mdir/Compiler/CompileCache.h"
#include "CompileCacheBuildInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/VCSRevision.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
#include <elf.h>
#include <link.h>
#include <map>
#include <mutex>
#include <sys/stat.h>
#include <vector>

#ifndef LLVM_REVISION
#define LLVM_REVISION "unknown"
#endif

using namespace llvm;

namespace {

/// An argument of the command line, or two (`-option value`), that named a
/// registered LLVM option, with the indices of its arguments in the
/// command line with its response files expanded.
struct RecordedArgument {
  const cl::Option *option = nullptr;
  unsigned first = 0, last = 0;
  std::string text;
};
std::mutex &getOptionsMutex() {
  static std::mutex mutex;
  return mutex;
}
std::vector<RecordedArgument> &getRecordedArguments() {
  static std::vector<RecordedArgument> recorded;
  return recorded;
}

/// The build ID of the loaded image that holds `address`, read from its
/// notes in memory (PT_NOTE, NT_GNU_BUILD_ID), and the name of its file
/// (empty for the executable).
struct Image {
  uintptr_t address = 0;
  bool found = false;
  std::string buildID;
  std::string file;
};

int visitImage(dl_phdr_info *info, size_t, void *data) {
  auto *image = static_cast<Image *>(data);
  bool holds = false;
  for (unsigned i = 0; i < info->dlpi_phnum && !holds; ++i) {
    const ElfW(Phdr) &header = info->dlpi_phdr[i];
    uintptr_t start = info->dlpi_addr + header.p_vaddr;
    holds = header.p_type == PT_LOAD && image->address >= start &&
            image->address < start + header.p_memsz;
  }
  if (!holds)
    return 0;
  image->found = true;
  image->file = info->dlpi_name ? info->dlpi_name : "";
  for (unsigned i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr) &header = info->dlpi_phdr[i];
    if (header.p_type != PT_NOTE)
      continue;
    const char *note =
        reinterpret_cast<const char *>(info->dlpi_addr + header.p_vaddr);
    const char *end = note + header.p_memsz;
    auto align = [](size_t size) { return (size + 3) & ~size_t(3); };
    while (note + sizeof(ElfW(Nhdr)) <= end) {
      ElfW(Nhdr) head;
      std::memcpy(&head, note, sizeof(head));
      const char *name = note + sizeof(head);
      const char *desc = name + align(head.n_namesz);
      const char *next = desc + align(head.n_descsz);
      if (next > end)
        break;
      if (head.n_type == NT_GNU_BUILD_ID && head.n_namesz == 4 &&
          std::memcmp(name, "GNU", 4) == 0 && head.n_descsz > 0) {
        image->buildID = toHex(
            ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(desc),
                              head.n_descsz),
            /*LowerCase=*/true);
        return 1;
      }
      note = next;
    }
  }
  return 1;
}

/// The identity of the image that holds this code: its build ID, else the
/// BLAKE3 of its file.
std::string getImageIdentity() {
  Image image;
  image.address = reinterpret_cast<uintptr_t>(&getImageIdentity);
  dl_iterate_phdr(visitImage, &image);
  if (!image.buildID.empty())
    return "build-id " + image.buildID;
  std::string file = image.file;
  if (file.empty())
    file = sys::fs::getMainExecutable(
        nullptr, reinterpret_cast<void *>(&getImageIdentity));
  if (auto hash = mdir::compiler::hashFile(file))
    return "file " + *hash;
  return "unknown";
}

/// The LLVM options set on the command line, each once, by its name, with
/// the arguments that set it; and the names of those set whose arguments
/// were not recorded. The scheduler is left out: every key takes the one in
/// force when it is taken.
struct SetOptions {
  std::map<std::string, std::vector<std::string>> recorded;
  std::vector<std::string> unrecorded;
};

SetOptions findSetOptions() {
  SetOptions set;
  std::lock_guard<std::mutex> lock(getOptionsMutex());
  const auto &arguments = getRecordedArguments();
  // An option is registered under its name and, with a parser of named
  // values (the passes of `mdir-opt`), under each value's name: each
  // option once, under its own name or else its first.
  std::map<const cl::Option *, std::string> options;
  for (const auto &entry : cl::getRegisteredOptions()) {
    const cl::Option *option = entry.second;
    if (!option->getNumOccurrences() || entry.first == "pre-RA-sched")
      continue;
    std::string &name = options[option];
    if (option->ArgStr == entry.first || name.empty())
      name = entry.first.str();
  }
  for (const auto &[option, name] : options) {
    std::vector<std::string> texts;
    for (const RecordedArgument &argument : arguments) {
      // A `cl::alias` passes its occurrences on to the option it names,
      // which takes the index of the alias's argument as its position.
      bool alias = argument.option->getNumOccurrences() == 0 &&
                   option->getPosition() >= argument.first &&
                   option->getPosition() <= argument.last;
      if (argument.option == option || alias)
        texts.push_back(argument.text);
    }
    if (texts.empty())
      set.unrecorded.push_back(name);
    else
      set.recorded[name + " x" +
                   std::to_string(option->getNumOccurrences())] =
          std::move(texts);
  }
  return set;
}

/// The LLVM options of the command line, sorted by name, each with the
/// arguments that gave it.
std::string describeOptions() {
  std::string text;
  for (const auto &[name, texts] : findSetOptions().recorded) {
    text += "cl-opt " + name + ":";
    for (const std::string &argument : texts)
      text += " " + argument;
    text += "\n";
  }
  return text;
}

} // namespace

void mdir::compiler::recordCommandLineOptions(int argc,
                                              const char *const *argv) {
  // The arguments as cl::ParseCommandLineOptions takes them: with each
  // `@file` replaced by the arguments it holds, which it reads in a copy of
  // its own. An expansion that fails records nothing; then the options set
  // are unrecorded and the cache is off (getUnrecordedOptions).
  BumpPtrAllocator allocator;
  StringSaver saver(allocator);
  SmallVector<const char *, 32> expanded(argv, argv + argc);
  if (!cl::ExpandResponseFiles(saver, cl::TokenizeGNUCommandLine, expanded))
    return;
  auto &options = cl::getRegisteredOptions();
  std::lock_guard<std::mutex> lock(getOptionsMutex());
  auto &recorded = getRecordedArguments();
  for (unsigned i = 1, e = expanded.size(); i < e; ++i) {
    StringRef argument(expanded[i]);
    if (argument == "--")
      break;
    if (!argument.starts_with("-"))
      continue;
    StringRef body = argument.ltrim('-');
    StringRef name = body.split('=').first;
    auto found = options.find(name);
    if (found == options.end())
      continue;
    RecordedArgument entry{found->second, i, i, argument.str()};
    // `-option value`: the value is the next argument.
    if (!body.contains('=') &&
        found->second->getValueExpectedFlag() == cl::ValueRequired &&
        i + 1 < e) {
      entry.last = ++i;
      entry.text += std::string(" ") + expanded[i];
    }
    recorded.push_back(std::move(entry));
  }
}

std::vector<std::string> mdir::compiler::getUnrecordedOptions() {
  return findSetOptions().unrecorded;
}

std::string mdir::compiler::getSchedulerInForce() {
  auto &options = cl::getRegisteredOptions();
  auto found = options.find("pre-RA-sched");
  if (found == options.end())
    return "none";
  // The option of SelectionDAGISel.cpp, of this type in LLVM 23.1.2; an
  // Option has no accessor of its value, so it is read through its type.
  using Scheduler = RegisterScheduler::FunctionPassCtor;
  using SchedulerOption =
      cl::opt<Scheduler, false, RegisterPassParser<RegisterScheduler>>;
  auto *option = static_cast<SchedulerOption *>(found->second);
  Scheduler value = option->getValue();
  auto &parser = option->getParser();
  for (unsigned i = 0, e = parser.getNumOptions(); i < e; ++i)
    if (static_cast<const cl::OptionValue<Scheduler> &>(
            parser.getOptionValue(i))
            .getValue() == value)
      return parser.getOption(i).str();
  return "unknown";
}

std::string mdir::compiler::getCacheSalt() {
  static const std::string fixed = [] {
    std::string text;
    raw_string_ostream os(text);
    os << "mdir-salt 1\n"
       << "image " << getImageIdentity() << '\n'
       << "mdir " << MDIR_CACHE_VERSION << " commit " << MDIR_GIT_COMMIT
       << " dirty " << MDIR_GIT_DIRTY << '\n'
       << "llvm " << LLVM_VERSION_STRING << ' ' << LLVM_REVISION << '\n'
       << describeOptions();
    return text;
  }();
  std::string salt = fixed;
  if (auto override = sys::Process::GetEnv("MDIR_COMPILE_CACHE_SALT_OVERRIDE");
      override && !override->empty())
    salt += "test-salt " + *override + "\n";
  return salt;
}

std::optional<std::string> mdir::compiler::hashFile(StringRef path) {
  static std::mutex mutex;
  static std::map<std::string, std::string> hashes;
  SmallString<256> name(path);
  struct stat status;
  if (::stat(name.c_str(), &status) != 0)
    return std::nullopt;
  // The time of change cannot be set back by a copy that keeps the time
  // of modification, as `cp -p` or `touch -r` does.
  std::string memo;
  raw_string_ostream os(memo);
  os << path << '\n'
     << status.st_dev << ' ' << status.st_ino << ' ' << status.st_size << ' '
     << status.st_mtim.tv_sec << '.' << status.st_mtim.tv_nsec << ' '
     << status.st_ctim.tv_sec << '.' << status.st_ctim.tv_nsec;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (auto found = hashes.find(memo); found != hashes.end())
      return found->second;
  }
  auto file = MemoryBuffer::getFile(path, /*IsText=*/false,
                                    /*RequiresNullTerminator=*/false);
  if (!file)
    return std::nullopt;
  std::string hash =
      toHex(BLAKE3::hash(arrayRefFromStringRef((*file)->getBuffer())),
            /*LowerCase=*/true);
  std::lock_guard<std::mutex> lock(mutex);
  return hashes[memo] = hash;
}
