#include "JITMemory.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <limits>
#include <map>

using namespace llvm;
using namespace mdir::compiler;
namespace {
Error invalid(const Twine &reason) {
  return createStringError(inconvertibleErrorCode(),
                           "JIT ownership: " + reason.str());
}
JITRange hull(ArrayRef<JITRange> ranges) {
  JITRange result{UINT64_MAX, 0};
  for (auto range : ranges) {
    result.begin = std::min(result.begin, range.begin);
    result.end = std::max(result.end, range.end);
  }
  return result;
}
std::mutex registryMutex;
std::map<const JITMemory *, JITRange> registry;
}
Error mdir::compiler::validateJITRanges(ArrayRef<JITRange> code,
                                       ArrayRef<JITRange> frames,
                                       ArrayRef<JITRange> live) {
  if (code.empty() || frames.empty())
    return invalid("missing executable sections or frame descriptions");
  for (auto range : code)
    if (range.begin >= range.end)
      return invalid("invalid executable range");
  for (auto frame : frames)
    if (!llvm::any_of(code, [&](JITRange range) { return range.contains(frame); }))
      return invalid("frame range is outside an allocated executable section");
  JITRange bounds = hull(code);
  for (auto other : live)
    if (bounds.begin < other.end && other.begin < bounds.end)
      return invalid("executable bounding interval overlaps a live object");
  return Error::success();
}
sys::MemoryBlock JITMemoryMapper::allocateMappedMemory(
    SectionMemoryManager::AllocationPurpose, size_t bytes,
    const sys::MemoryBlock *near, unsigned flags, std::error_code &error) {
  auto block = sys::Memory::allocateMappedMemory(bytes, near, flags, error);
  if (!error)
    allocations.push_back({reinterpret_cast<uintptr_t>(block.base()),
                           reinterpret_cast<uintptr_t>(block.base()) +
                               block.allocatedSize()});
  return block;
}
std::error_code JITMemoryMapper::protectMappedMemory(
    const sys::MemoryBlock &block, unsigned flags) {
  return sys::Memory::protectMappedMemory(block, flags);
}
std::error_code JITMemoryMapper::releaseMappedMemory(sys::MemoryBlock &block) {
  // Never free code or frame bytes while the unwinder can still reach them.
  if (registered)
    report_fatal_error("JIT ownership: release before frame deregistration");
  return sys::Memory::releaseMappedMemory(block);
}
JITMemory::JITMemory(Triple triple)
    : SectionMemoryManager(static_cast<JITMemoryMapper *>(this), true),
      triple(std::move(triple)) {}
JITMemory::~JITMemory() { deregisterEHFrames(); }
uint8_t *JITMemory::allocateCodeSection(uintptr_t size, unsigned alignment,
                                       unsigned id, StringRef name) {
  if (state != State::Linking)
    failure = "allocation after object finalization";
  auto *address = SectionMemoryManager::allocateCodeSection(size, alignment, id, name);
  if (size)
    code.push_back({reinterpret_cast<uintptr_t>(address),
                    reinterpret_cast<uintptr_t>(address) + size});
  return address;
}
uint8_t *JITMemory::allocateDataSection(uintptr_t size, unsigned alignment,
                                       unsigned id, StringRef name, bool readOnly) {
  if (state != State::Linking)
    failure = "allocation after object finalization";
  auto *address = SectionMemoryManager::allocateDataSection(size, alignment, id, name, readOnly);
  if (size)
    data.push_back({reinterpret_cast<uintptr_t>(address),
                    reinterpret_cast<uintptr_t>(address) + size});
  return address;
}
void JITMemory::registerEHFrames(uint8_t *address, uint64_t load, size_t size) {
  // RuntimeDyld requests registration before finalizeMemory. Delay publishing
  // to the process unwinder until the relocated bytes have been checked.
  if (state != State::Linking)
    failure = "frame registration after object finalization";
  pending.push_back({address, load, size});
}
bool JITMemory::finalizeMemory(std::string *message) {
  auto reject = [&](const Twine &reason) {
    state = State::Rejected;
    if (message)
      *message = "JIT ownership: " + reason.str();
    return true;
  };
  if (state != State::Linking)
    return reject("object finalized more than once");
  if (!failure.empty())
    return reject(failure);
  // ReserveAllocationSpace is an optimization supplied by LLVM, not the
  // proof: verify that the actual code hull lies in one owned mapping.
  if (code.empty() || !llvm::any_of(allocations, [&](JITRange allocation) {
        return allocation.contains(hull(code));
      }))
    return reject("executable sections do not share an owned allocation");
  if (pending.size() != 1)
    return reject("expected one unwind table per object");
  std::vector<JITRange> frames;
  for (auto frame : pending) {
    uintptr_t address = reinterpret_cast<uintptr_t>(frame.address);
    if (!frame.size || frame.load != address ||
        frame.size > UINT64_MAX - address ||
        !llvm::any_of(data, [&](JITRange section) {
          return section.contains({address, address + frame.size});
        }))
      return reject("unwind bytes are outside an allocated data section");
    DWARFDebugFrame parsed(triple.getArch(), true, address);
    if (auto error = parsed.parse(DWARFDataExtractor(
            StringRef(reinterpret_cast<const char *>(frame.address), frame.size),
            triple.isLittleEndian(), triple.isArch64Bit() ? 8 : 4)))
      return reject("cannot parse relocated unwind bytes: " + toString(std::move(error)));
    for (const auto &entry : parsed.entries())
      if (auto *fde = dyn_cast<dwarf::FDE>(&entry)) {
        uint64_t begin = fde->getInitialLocation(), size = fde->getAddressRange();
        if (!fde->getLinkedCIE() || !size || size > UINT64_MAX - begin)
          return reject("invalid frame description");
        frames.push_back({begin, begin + size});
      }
  }
  std::lock_guard<std::mutex> lock(registryMutex);
  std::vector<JITRange> live;
  for (auto &entry : registry)
    live.push_back(entry.second);
  if (auto error = validateJITRanges(code, frames, live))
    return reject(toString(std::move(error)));
  if (SectionMemoryManager::finalizeMemory(message)) {
    state = State::Rejected;
    return true;
  }
  for (auto frame : pending)
    SectionMemoryManager::registerEHFrames(frame.address, frame.load, frame.size);
  registry.emplace(this, hull(code));
  registered = true;
  state = State::Registered;
  return false;
}
void JITMemory::deregisterEHFrames() {
  std::lock_guard<std::mutex> lock(registryMutex);
  // ORC removal and the destructor can both request cleanup. It is idempotent.
  SectionMemoryManager::deregisterEHFrames();
  registry.erase(this);
  registered = false;
  pending.clear();
  state = State::Released;
}
