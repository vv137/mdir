// Validate the transport of positions and signed integer contributions using
// the topology's actual members. The integer payload is a diagnostic of
// the input words. It is deliberately independent of a physical force scale.
#include "partition.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(result));
    std::exit(1);
  }
}

template<class T> class Buffer {
public:
  explicit Buffer(size_t size) {
    check(cudaMalloc(reinterpret_cast<void **>(&data), sizeof(T) * std::max(size, size_t(1))));
  }
  explicit Buffer(const std::vector<T> &values) : Buffer(values.size()) {
    if (!values.empty())
      check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ~Buffer() { check(cudaFree(data)); }
  T *data;
};

struct Block {
  unsigned group, firstAtom, count, firstRun, endRun;
};

// Consecutive instructions of equal arity have equal stride in words.
struct Run {
  unsigned arity, firstWord, count;
};

__host__ __device__ uint64_t mix(uint64_t hash, uint64_t word) {
  return (hash ^ word) * 1099511628211ULL;
}

__host__ __device__ int contribution(uint64_t hash, unsigned slot,
                                     unsigned component) {
  uint64_t h = mix(hash, 3 * slot + component + 1);
  return static_cast<int>((h ^ (h >> 32)) & 0x3ffff) - 131072;
}

// Each block stages the positions once, then processes the term runs in
// parallel. One lane flushes each owned atom after the shared-memory barrier.
__global__ void verifyUnits(const Block *blocks, const Run *runs,
                            const unsigned *imports, const unsigned *words,
                            const double *positions, unsigned width,
                            unsigned long long *output, unsigned *energyOwners) {
  Block block = blocks[blockIdx.x];
  extern __shared__ unsigned long long shared[];
  auto *forces = shared + 3 * block.count;
  for (unsigned i = threadIdx.x; i < block.count; i += blockDim.x) {
    unsigned atom = imports[block.firstAtom + i];
    for (unsigned c = 0; c < 3; ++c) {
      shared[3 * i + c] = __double_as_longlong(positions[3 * atom + c]);
      forces[3 * i + c] = 0;
    }
  }
  __syncthreads();
  for (unsigned r = block.firstRun; r < block.endRun; ++r) {
    Run run = runs[r];
    for (unsigned n = threadIdx.x; n < run.count; n += blockDim.x) {
      unsigned p = run.firstWord + n * (run.arity > 3 ? 3 : 2);
      unsigned term = words[p];
      uint64_t slots = words[p + 1];
      if (run.arity > 3)
        slots |= uint64_t(words[p + 2]) << 32;
      uint64_t hash = term + 1;
      for (unsigned j = 0; j < run.arity; ++j) {
        unsigned local = (slots >> (10 * j)) & 1023;
        for (unsigned c = 0; c < 3; ++c)
          hash = mix(hash, shared[3 * local + c]);
      }
      for (unsigned j = 0; j < run.arity; ++j) {
        unsigned local = (slots >> (10 * j)) & 1023;
        if (imports[block.firstAtom + local] / width != block.group)
          continue;
        if (j == 0)
          atomicAdd(energyOwners + term, 1u);
        for (unsigned c = 0; c < 3; ++c)
          atomicAdd(forces + 3 * local + c,
                    static_cast<unsigned long long>(contribution(hash, j, c)));
      }
    }
  }
  __syncthreads();
  for (unsigned i = threadIdx.x; i < block.count; i += blockDim.x) {
    unsigned atom = imports[block.firstAtom + i];
    if (atom / width == block.group)
      for (unsigned c = 0; c < 3; ++c)
        atomicAdd(output + 3 * atom + c, forces[3 * i + c]);
  }
}
} // namespace

void bonded::checkOnDevice(const std::vector<Term> &terms,
                           const std::vector<Unit> &units,
                           const std::vector<double> &positions, unsigned width) {
  std::vector<Block> blocks;
  std::vector<Run> runs;
  std::vector<unsigned> imports, words;
  size_t maxAtoms = 0;
  for (unsigned u = 0; u < units.size(); ++u) {
    const Unit &unit = units[u];
    if (unit.terms.empty())
      continue;
    Block b{u, unsigned(imports.size()), unsigned(unit.atoms.size()),
            unsigned(runs.size()), 0};
    unsigned cursor = words.size();
    for (unsigned t : unit.terms) {
      unsigned arity = terms[t].arity;
      if (runs.size() == b.firstRun || runs.back().arity != arity)
        runs.push_back({arity, cursor, 0});
      ++runs.back().count;
      cursor += arity > 3 ? 3 : 2;
    }
    b.endRun = runs.size();
    blocks.push_back(b);
    imports.insert(imports.end(), unit.atoms.begin(), unit.atoms.end());
    words.insert(words.end(), unit.instructions.begin(), unit.instructions.end());
    maxAtoms = std::max(maxAtoms, unit.atoms.size());
  }
  if (blocks.empty())
    return;
  Buffer<Block> deviceBlocks(blocks);
  Buffer<Run> deviceRuns(runs);
  Buffer<unsigned> deviceImports(imports), deviceWords(words), deviceOwners(terms.size());
  Buffer<double> devicePositions(positions);
  Buffer<unsigned long long> deviceForces(positions.size());
  check(cudaMemset(deviceOwners.data, 0, terms.size() * sizeof(unsigned)));
  check(cudaMemset(deviceForces.data, 0, positions.size() * sizeof(unsigned long long)));
  verifyUnits<<<blocks.size(), 128, 6 * sizeof(uint64_t) * maxAtoms>>>(
      deviceBlocks.data, deviceRuns.data, deviceImports.data, deviceWords.data,
      devicePositions.data, width, deviceForces.data, deviceOwners.data);
  check(cudaGetLastError());
  std::vector<unsigned long long> actual(positions.size()), expected(positions.size());
  std::vector<unsigned> owners(terms.size());
  check(cudaMemcpy(actual.data(), deviceForces.data, actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost));
  check(cudaMemcpy(owners.data(), deviceOwners.data, owners.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));

  // The oracle reads the original global positions and term table directly.
  for (unsigned t = 0; t < terms.size(); ++t) {
    const auto &term = terms[t];
    uint64_t hash = t + 1;
    for (unsigned j = 0; j < term.arity; ++j)
      for (unsigned c = 0; c < 3; ++c) {
        uint64_t bits;
        std::memcpy(&bits, &positions[3 * term.atoms[j] + c], sizeof(bits));
        hash = mix(hash, bits);
      }
    for (unsigned j = 0; j < term.arity; ++j)
      for (unsigned c = 0; c < 3; ++c)
        expected[3 * term.atoms[j] + c] += contribution(hash, j, c);
  }
  if (expected != actual ||
      !std::all_of(owners.begin(), owners.end(), [](unsigned n) { return n == 1; })) {
    std::fprintf(stderr, "device work-unit result differs from direct host oracle\n");
    std::exit(1);
  }
  std::fprintf(stderr, "device integer oracle passed: width=%u terms=%zu components=%zu\n",
               width, terms.size(), positions.size());
}
