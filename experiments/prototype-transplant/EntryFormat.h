// Diagnostic I/O conversion only. No search or distance predicate belongs here.
#ifndef MDIR_EXPERIMENT_PROTOTYPE_ENTRY_FORMAT_H
#define MDIR_EXPERIMENT_PROTOTYPE_ENTRY_FORMAT_H
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>
namespace prototype_io {
using Image = std::array<int64_t, 3>;
struct PhysicalPair {
  int32_t kind, i, j;
  Image image;
  auto key() const { return std::tie(kind, i, j, image); }
  bool operator<(const PhysicalPair &b) const { return key() < b.key(); }
  bool operator==(const PhysicalPair &b) const { return key() == b.key(); }
};
struct NativeGroup {
  int32_t first;
  std::vector<int32_t> members;
};
struct NativeEntry {
  int32_t kind, row, partner;
  uint32_t image, mask;
  int32_t reserved = 0;
  auto key() const { return std::tie(kind, row, partner, image, mask, reserved); }
  bool operator<(const NativeEntry &b) const { return key() < b.key(); }
  bool operator==(const NativeEntry &b) const { return key() == b.key(); }
};
struct NativeBasis {
  int32_t waters;
  std::vector<int32_t> place;
  std::vector<NativeGroup> waterGroups, atomGroups;
  std::vector<int32_t> groupOf, bitOf;
  /// Native general masks partition the first group by its lattice frame.
  /// This is an internal regroup key, not a public payload field.
  std::vector<Image> frames;
  void validate() {
    if (waters < 0 || static_cast<size_t>(waters) > place.size())
      throw std::runtime_error("bad native water domain");
    if (frames.empty()) frames.assign(place.size(),Image{0,0,0});
    if (frames.size()!=place.size())throw std::runtime_error("native frame domain mismatch");
    groupOf.assign(place.size(), -1);
    bitOf.assign(place.size(), -1);
    std::vector<int32_t> used(place.size(), 0);
    for (int32_t p : place) {
      if (p < 0 || static_cast<size_t>(p) >= used.size() || used[p]++)
        throw std::runtime_error("native places are not a permutation");
    }
    for (unsigned kind = 0; kind < 2; ++kind) {
      const auto &groups = kind ? atomGroups : waterGroups;
      for (size_t g = 0; g < groups.size(); ++g) {
        const auto &group = groups[g];
        if (group.members.empty() || group.members.size() > 16)
          throw std::runtime_error("native group length outside1..16");
        for (size_t bit = 0; bit < group.members.size(); ++bit) {
          int32_t unit = group.members[bit];
          if (unit < 0 || static_cast<size_t>(unit) >= place.size() ||
              (unit >= waters) != bool(kind) || groupOf[unit] != -1 ||
              place[unit] != group.first + static_cast<int32_t>(bit))
            throw std::runtime_error("inconsistent native group/member/place");
          groupOf[unit] = static_cast<int32_t>(g);
          bitOf[unit] = static_cast<int32_t>(bit);
        }
      }
    }
    if (std::find(groupOf.begin(), groupOf.end(), -1) != groupOf.end())
      throw std::runtime_error("native group coverage incomplete");
  }
  int32_t row(int32_t kind, int32_t unit) const {
    int32_t offset = kind == 0 ? 0 : static_cast<int32_t>(waterGroups.size());
    if (kind == 2) offset += static_cast<int32_t>(atomGroups.size());
    return offset + groupOf.at(unit);
  }
};
struct PrototypeBasis {
  int32_t waters;
  /// Original unit IDs, padded to16 per prototype group; -1 means padding.
  std::vector<int32_t> waterOrder, atomOrder;
  /// Actual retained s2_keys shifts, indexed by original logical unit ID.
  std::vector<Image> shifts;
  void validate() const {
    if (waters < 0 || static_cast<size_t>(waters) > shifts.size() ||
        waterOrder.size() % 16 || atomOrder.size() % 16)
      throw std::runtime_error("invalid padded prototype order shape");
    std::vector<int> seen(shifts.size(),0);
    for (unsigned kind=0;kind<2;++kind) {
      const auto &order=kind?atomOrder:waterOrder;
      for (int32_t unit:order) if (unit!=-1) {
        if (unit<0 || static_cast<size_t>(unit)>=seen.size() ||
            (unit>=waters)!=bool(kind) || seen[unit]++)
          throw std::runtime_error("invalid prototype order/member class");
      }
    }
    if (std::find(seen.begin(),seen.end(),0)!=seen.end())
      throw std::runtime_error("prototype order omits a unit");
  }
};
struct PrototypeEntry {
  int32_t kind, group, partnerSlot;
  uint32_t packed;
};
inline uint32_t nativeImage(const Image &image) {
  uint32_t result = 0;
  for (unsigned a = 0; a < 3; ++a) {
    if (image[a] < -512 || image[a] >= 512)
      throw std::runtime_error("native image domain exceeded");
    result |= static_cast<uint32_t>(image[a] + 512) << (10 * a);
  }
  return result;
}
inline Image decodeNativeImage(uint32_t code) {
  if (code >> 30) throw std::runtime_error("invalid native image bits");
  Image image;
  for (unsigned a = 0; a < 3; ++a) image[a] = ((code >> (10 * a)) & 1023) - int64_t(512);
  return image;
}
inline Image prototypeImage(uint32_t packed) {
  if (packed >> 28) throw std::runtime_error("invalid prototype image bits");
  Image image;
  for (unsigned a = 0; a < 3; ++a) image[a] = ((packed >> (16 + 4 * a)) & 15) - int64_t(4);
  return image;
}
inline PhysicalPair orient(PhysicalPair pair, const NativeBasis &native) {
  if (pair.kind < 0 || pair.kind > 2 || pair.i < 0 || pair.j < 0 ||
      static_cast<size_t>(pair.i) >= native.place.size() ||
      static_cast<size_t>(pair.j) >= native.place.size())
    throw std::runtime_error("physical pair domain exceeded");
  if ((pair.kind == 0 && (pair.i >= native.waters || pair.j >= native.waters)) ||
      (pair.kind == 1 && (pair.i < native.waters || pair.j >= native.waters)) ||
      (pair.kind == 2 && (pair.i < native.waters || pair.j < native.waters)))
    throw std::runtime_error("class ownership mismatch");
  if (pair.kind != 1) {
    if (pair.i == pair.j) throw std::runtime_error("self pair in half list");
    if (native.place[pair.i] > native.place[pair.j]) {
      std::swap(pair.i, pair.j);
      for (auto &v : pair.image) v = -v;
    }
  }
  return pair;
}
inline std::vector<PhysicalPair> expandPrototype(const std::vector<PrototypeEntry> &entries,
    const PrototypeBasis &proto, const NativeBasis &native) {
  proto.validate();
  if (proto.waters != native.waters || proto.shifts.size() != native.place.size())
    throw std::runtime_error("prototype/native unit domains disagree");
  std::vector<PhysicalPair> result;
  if (entries.size() > result.max_size() / 16)
    throw std::runtime_error("physical expansion capacity overflow");
  result.reserve(entries.size() * 16);
  for (const auto &entry : entries) {
    if (entry.kind < 0 || entry.kind > 2 || entry.group < 0 || entry.partnerSlot < 0)
      throw std::runtime_error("invalid prototype entry index");
    const auto &first = entry.kind == 0 ? proto.waterOrder : proto.atomOrder;
    const auto &second = entry.kind == 2 ? proto.atomOrder : proto.waterOrder;
    if (static_cast<size_t>(entry.partnerSlot) >= second.size())
      throw std::runtime_error("prototype partner slot out of bounds");
    int32_t j = second[entry.partnerSlot];
    if (j < 0 || static_cast<size_t>(j) >= proto.shifts.size())
      throw std::runtime_error("prototype partner is padding");
    uint32_t mask = entry.packed & 65535;
    if (!mask) throw std::runtime_error("empty prototype entry");
    Image code = prototypeImage(entry.packed);
    for (unsigned bit = 0; bit < 16; ++bit) if (mask & (1u << bit)) {
      size_t slot = static_cast<size_t>(entry.group) * 16 + bit;
      if (slot >= first.size()) throw std::runtime_error("prototype first group out of bounds");
      int32_t i = first[slot];
      if (i < 0 || static_cast<size_t>(i) >= proto.shifts.size())
        throw std::runtime_error("prototype mask addresses padding");
      Image image;
      for (unsigned a = 0; a < 3; ++a)
        image[a] = code[a] - proto.shifts[i][a] + proto.shifts[j][a];
      result.push_back(orient({entry.kind, i, j, image}, native));
    }
  }
  std::sort(result.begin(), result.end());
  if (std::adjacent_find(result.begin(), result.end()) != result.end())
    throw std::runtime_error("duplicate physical pair in prototype list");
  return result;
}
inline std::vector<NativeEntry> regroup(const std::vector<PhysicalPair> &pairs,
                                      const NativeBasis &native) {
  using Piece = std::tuple<int32_t, int32_t, int32_t, uint32_t, Image, uint32_t>;
  std::vector<Piece> pieces;
  pieces.reserve(pairs.size());
  for (const auto &given : pairs) {
    auto pair = orient(given, native);
    pieces.emplace_back(pair.kind, native.row(pair.kind, pair.i), pair.j,
        nativeImage(pair.image), native.frames.at(pair.i), 1u << native.bitOf.at(pair.i));
  }
  std::sort(pieces.begin(), pieces.end());
  std::vector<NativeEntry> result;
  Image previousFrame{};
  for (auto [kind, row, partner, image, frame, bit] : pieces) {
    if (!result.empty() && std::tie(result.back().kind, result.back().row,
          result.back().partner, result.back().image) == std::tie(kind, row, partner, image) && previousFrame==frame) {
      if (result.back().mask & bit) throw std::runtime_error("duplicate physical bit while regrouping");
      result.back().mask |= bit;
    } else result.push_back({kind, row, partner, image, bit, 0});
    previousFrame=frame;
  }
  std::sort(result.begin(), result.end());
  return result;
}
inline std::vector<PhysicalPair> expandNative(const std::vector<NativeEntry> &entries,
                                            const NativeBasis &native) {
  std::vector<PhysicalPair> result;
  for (const auto &entry : entries) {
    if (entry.reserved || entry.kind < 0 || entry.kind > 2 || !entry.mask || entry.mask >> 16)
      throw std::runtime_error("invalid native entry fields");
    int32_t offset = entry.kind == 0 ? 0 : static_cast<int32_t>(native.waterGroups.size());
    if (entry.kind == 2) offset += static_cast<int32_t>(native.atomGroups.size());
    const auto &groups = entry.kind == 0 ? native.waterGroups : native.atomGroups;
    int64_t group = int64_t(entry.row) - offset;
    if (group < 0 || static_cast<size_t>(group) >= groups.size())
      throw std::runtime_error("native row outside class");
    auto image = decodeNativeImage(entry.image);
    for (unsigned bit = 0; bit < 16; ++bit) if (entry.mask & (1u << bit)) {
      if (bit >= groups[group].members.size()) throw std::runtime_error("native padded I bit");
      result.push_back(orient({entry.kind, groups[group].members[bit], entry.partner, image}, native));
    }
  }
  std::sort(result.begin(), result.end());
  if (std::adjacent_find(result.begin(), result.end()) != result.end())
    throw std::runtime_error("duplicate physical pair in native list");
  return result;
}
} // namespace prototype_io
#endif
