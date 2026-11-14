// The layout contract of a neighbor structure (D[neighbors-contract]).
//
// What a build guarantees about the lists of a structure, on two axes:
// multiplicity (the traversal, the images, the encoding of the excluded
// pairs) and organization (none, grouped by a key, sorted within a group
// by a key, stable), the values of D409. The same values state what a
// consumer requires, and a requirement `r` is met by a guarantee `g` when
// `r ⊑ g` on every axis.

#ifndef MDIR_DIALECT_MDRT_NEIGHBORSCONTRACT_H
#define MDIR_DIALECT_MDRT_NEIGHBORSCONTRACT_H

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>
#include <string>

namespace mdir {
namespace mdrt {

struct NeighborsContract {
  /// Which pairs a list holds: both orders, or each pair once.
  enum class Traversal : uint8_t { Directed, Unique };
  /// The displacement of an entry: the minimum image in the cell, or the
  /// plain difference of positions that carry their images (D290).
  enum class Images : uint8_t { Minimum, Explicit };
  /// How the excluded pairs are left out. `Masked`: an entry may stand for
  /// an excluded pair (the particle itself, on a device's matrix), which a
  /// consumer skips. `Removed`: no entry stands for one. `Masked ⊑ Removed`.
  enum class Exclusions : uint8_t { Masked, Removed };
  /// The organization of the entries, ordered by strength.
  enum class Organization : uint8_t { None, GroupedBy, SortedWithin, Stable };
  /// The key of a grouping: the row of one center particle, or a group of
  /// the places of a build of groups.
  enum class Group : uint8_t { None, Owner, Group };
  /// The key of an order within a group: the partner, or the order in
  /// which the build's search finds the entries. Both are total, so a
  /// list sorted by either is stable.
  enum class Order : uint8_t { None, Item, Search };

  Traversal traversal = Traversal::Directed;
  Images images = Images::Minimum;
  Exclusions exclusions = Exclusions::Masked;
  Organization organization = Organization::None;
  Group group = Group::None;
  Order order = Order::None;

  /// What a matrix guarantees, on either target: the rows of the centers,
  /// in the order of the search; an excluded pair may be the particle
  /// itself (a device's matrix), so the exclusions are `masked`.
  static NeighborsContract matrix(Images images = Images::Minimum);
  /// What a structure of groups guarantees, and its dual list (D89, D114).
  static NeighborsContract groups();

  /// The organization `organization` with its keys, normalized: a sort by
  /// a total key is stable (D409).
  static NeighborsContract withOrganization(NeighborsContract base,
                                            Organization organization,
                                            Group group, Order order);

  /// Returns true if the value is normalized: no key without its kind and
  /// no kind without its keys, and `sorted_within` with no total key.
  bool isNormalized() const;

  /// Returns true if this guarantee meets the requirement `required`.
  bool meets(const NeighborsContract &required) const;
  /// The axis on which this guarantee does not meet `required`, in words,
  /// or nothing if it meets it.
  std::optional<std::string> whyNot(const NeighborsContract &required) const;

  /// The least requirement that meets both, if the two are comparable on
  /// the organization (D409: different keys are not).
  static std::optional<NeighborsContract>
  joinOrganization(const NeighborsContract &a, const NeighborsContract &b);

  bool operator==(const NeighborsContract &other) const {
    return traversal == other.traversal && images == other.images &&
           exclusions == other.exclusions &&
           organization == other.organization && group == other.group &&
           order == other.order;
  }
  bool operator!=(const NeighborsContract &other) const {
    return !(*this == other);
  }

  /// The four words of the type, `directed, minimum, masked,
  /// stable(owner, search)`.
  void print(llvm::raw_ostream &os) const;
  std::string str() const;
  /// The organization alone, `stable(owner, search)`.
  void printOrganization(llvm::raw_ostream &os) const;
};

inline llvm::hash_code hash_value(const NeighborsContract &contract) {
  return llvm::hash_combine(contract.traversal, contract.images,
                            contract.exclusions, contract.organization,
                            contract.group, contract.order);
}

llvm::StringRef stringify(NeighborsContract::Traversal value);
llvm::StringRef stringify(NeighborsContract::Images value);
llvm::StringRef stringify(NeighborsContract::Exclusions value);
llvm::StringRef stringify(NeighborsContract::Group value);
llvm::StringRef stringify(NeighborsContract::Order value);

} // namespace mdrt
} // namespace mdir

#endif // MDIR_DIALECT_MDRT_NEIGHBORSCONTRACT_H
