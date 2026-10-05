#ifndef MDIR_COMPILER_JITMEMORY_H
#define MDIR_COMPILER_JITMEMORY_H

#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/TargetParser/Triple.h"
#include <mutex>
#include <vector>

namespace mdir::compiler {
struct JITRange {
  uint64_t begin, end;
  bool contains(JITRange other) const {
    return begin <= other.begin && other.begin < other.end && other.end <= end;
  }
};
/// Shared by production and deterministic boundary tests.
llvm::Error validateJITRanges(llvm::ArrayRef<JITRange> code,
                              llvm::ArrayRef<JITRange> frames,
                              llvm::ArrayRef<JITRange> live);

/// Member ordering is deliberate: the mapper must outlive the memory manager.
class JITMemoryMapper : public llvm::SectionMemoryManager::MemoryMapper {
public:
  llvm::sys::MemoryBlock allocateMappedMemory(
      llvm::SectionMemoryManager::AllocationPurpose purpose, size_t bytes,
      const llvm::sys::MemoryBlock *near, unsigned flags,
      std::error_code &error) override;
  std::error_code protectMappedMemory(const llvm::sys::MemoryBlock &block,
                                      unsigned flags) override;
  std::error_code releaseMappedMemory(llvm::sys::MemoryBlock &block) override;
  std::vector<JITRange> allocations;
  bool registered = false;
};
class JITMemory final : private JITMemoryMapper,
                        public llvm::SectionMemoryManager {
public:
  explicit JITMemory(llvm::Triple triple);
  ~JITMemory() override;
  uint8_t *allocateCodeSection(uintptr_t size, unsigned alignment, unsigned id,
                               llvm::StringRef name) override;
  uint8_t *allocateDataSection(uintptr_t size, unsigned alignment, unsigned id,
                               llvm::StringRef name, bool readOnly) override;
  void registerEHFrames(uint8_t *address, uint64_t load, size_t size) override;
  void deregisterEHFrames() override;
  bool finalizeMemory(std::string *message) override;
private:
  enum class State { Linking, Registered, Released, Rejected };
  State state = State::Linking;
  llvm::Triple triple;
  std::vector<JITRange> code, data;
  struct Frame { uint8_t *address; uint64_t load; size_t size; };
  std::vector<Frame> pending;
  std::string failure;
};
} // namespace mdir::compiler
#endif
