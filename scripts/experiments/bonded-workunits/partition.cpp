// Topology-order valence units following Cerutti et al. (2024), Sec. IV B.
// The driver reads the same constrained topology as a simulation.
#include "partition.h"
#include "mdir/Driver/System.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

using namespace mdir::driver;

namespace {
using bonded::Term;
using bonded::Unit;

void measure(const std::vector<Term> &terms, const std::vector<double> &positions,
             unsigned width) {
  size_t particles = positions.size() / 3;
  std::vector<Unit> units((particles + width - 1) / width);
  size_t slots = 0;
  for (unsigned t = 0; t < terms.size(); ++t) {
    const auto &term = terms[t];
    slots += term.arity;
    std::set<unsigned> owners;
    for (unsigned j = 0; j < term.arity; ++j)
      owners.insert(term.atoms[j] / width);
    for (auto owner : owners) {
      auto &unit = units[owner];
      unit.terms.push_back(t);
      unit.atoms.insert(unit.atoms.end(), term.atoms.begin(),
                        term.atoms.begin() + term.arity);
    }
  }
  size_t active = 0, copies = 0, imports = 0, exports = 0, maxAtoms = 0;
  size_t instructionBytes = 0;
  std::vector<unsigned> seenSlots(terms.size()), energyOwners(terms.size());
  for (unsigned u = 0; u < units.size(); ++u) {
    auto &unit = units[u];
    if (unit.terms.empty())
      continue;
    ++active;
    auto &atoms = unit.atoms;
    std::sort(atoms.begin(), atoms.end());
    atoms.erase(std::unique(atoms.begin(), atoms.end()), atoms.end());
    if (atoms.size() > 1024)
      llvm::report_fatal_error("local IDs exceed ten bits");
    imports += atoms.size();
    maxAtoms = std::max(maxAtoms, atoms.size());
    exports += std::count_if(atoms.begin(), atoms.end(),
                             [&](unsigned a) { return a / width == u; });
    copies += unit.terms.size();
    for (unsigned t : unit.terms) {
      const auto &term = terms[t];
      uint64_t members = 0;
      for (unsigned j = 0; j < term.arity; ++j) {
        auto local = std::lower_bound(atoms.begin(), atoms.end(), term.atoms[j]);
        members |= uint64_t(local - atoms.begin()) << (10 * j);
      }
      unit.instructions.push_back(t);
      unit.instructions.push_back(static_cast<uint32_t>(members));
      if (term.arity > 3)
        unit.instructions.push_back(members >> 32);
    }
    instructionBytes += sizeof(uint32_t) * unit.instructions.size();

    // Decode with only the import list, the instruction stream and the
    // original term table. Check ownership for each slot of each term.
    size_t cursor = 0;
    while (cursor < unit.instructions.size()) {
      unsigned t = unit.instructions.at(cursor++);
      const auto &term = terms.at(t);
      uint64_t members = unit.instructions.at(cursor++);
      if (term.arity > 3)
        members |= uint64_t(unit.instructions.at(cursor++)) << 32;
      for (unsigned j = 0; j < term.arity; ++j) {
        unsigned a = atoms.at((members >> (10 * j)) & 1023);
        if (a != term.atoms[j])
          llvm::report_fatal_error("member round trip failed");
        if (a / width != u)
          continue;
        unsigned slot = 1u << j;
        if (seenSlots[t] & slot)
          llvm::report_fatal_error("two units write one force slot");
        seenSlots[t] |= slot;
        if (j == 0)
          ++energyOwners[t];
      }
    }
  }
  for (size_t t = 0; t < terms.size(); ++t)
    if (seenSlots[t] != (1u << terms[t].arity) - 1 || energyOwners[t] != 1)
      llvm::report_fatal_error("work unit ownership mismatch");
#ifdef MDIR_CHECK_BONDED_DEVICE
  bonded::checkOnDevice(terms, units, positions, width);
#endif
  std::printf("%u,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n", width,
              terms.size(), slots, active, copies, imports, exports, maxAtoms,
              4 * slots, instructionBytes);
}

// A fused integration kernel would have to move these groups together.
void countSplitConstraints(const Topology &top, unsigned width) {
  size_t splitWater = 0, splitShake = 0;
  for (const auto &water : top.settles)
    splitWater += water.oxygen / width != (water.oxygen + 2) / width;
  for (const auto &group : top.shakes) {
    unsigned unit = group.center / width;
    splitShake += std::any_of(group.hydrogens.begin(), group.hydrogens.end(),
                              [=](unsigned h) { return h / width != unit; });
  }
  std::fprintf(stderr,
               "width=%u split_settle=%zu/%zu split_shake=%zu/%zu\n",
               width, splitWater, top.settles.size(), splitShake,
               top.shakes.size());
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: bonded-partition control.toml\n");
    return 2;
  }
  llvm::ExitOnError check;
  auto control = check(readControl(argv[1]));
  auto system = check(readSystem(control));
  if (!system.topology)
    llvm::report_fatal_error("expected a molecular topology");
  const auto &top = *system.topology;
  std::vector<Term> terms;
  for (auto &b : top.bonds)
    terms.push_back({{b.i, b.j}, 2});
  for (auto &a : top.angles)
    terms.push_back({{a.i, a.j, a.k}, 3});
  for (auto &d : top.dihedrals)
    terms.push_back({{d.i, d.j, d.k, d.l}, 4});
  for (auto &m : top.cmaps)
    terms.push_back({{m.i, m.j, m.k, m.l, m.m}, 5});
  for (auto &p : top.pairs)
    terms.push_back({{p.i, p.j}, 2});
  std::fprintf(stderr, "atoms=%zu bonds=%zu angles=%zu dihedrals=%zu cmaps=%zu pairs14=%zu exclusions=%zu\n",
               system.getNumParticles(), top.bonds.size(), top.angles.size(),
               top.dihedrals.size(), top.cmaps.size(), top.pairs.size(),
               top.exclusions.size());
  std::printf("scope,width,terms,slots,units,evaluations,imports,owned,max_local,member_bytes,packed_bytes\n");
  for (unsigned width : {64, 128, 256, 512}) {
    countSplitConstraints(top, width);
    std::printf("valence,");
    measure(terms, system.positions, width);
  }
  for (auto [a, b] : top.exclusions)
    terms.push_back({{a, b}, 2});
  for (unsigned width : {64, 128, 256, 512}) {
    std::printf("with_exclusions,");
    measure(terms, system.positions, width);
  }
}
