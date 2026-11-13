// The shared analysis of the collections of md_alg (D[md-alg-analysis],
// #866): for each buffer that an md_alg op produces, its ordering, access
// pattern, cardinality, effects and aliases, lifetime, and determinism
// requirement, and the order that its consumers require, propagated back
// to its producer. Every collection-level transformation of
// `md-alg-transform-collections` reads it instead of judging legality on
// its own. Stage A of the maintainer's item 3: on memrefs, through the
// effects that the ops declare.
//
// See docs/decisions/md-alg-analysis.md for the lattice and the
// conservative rules.

#ifndef MDIR_DIALECT_MDALG_COLLECTIONANALYSIS_H
#define MDIR_DIALECT_MDALG_COLLECTIONANALYSIS_H

#include "mdir/Dialect/MDAlg/MDAlgOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace mlir {
class AsmState;
} // namespace mlir

namespace mdir::md_alg {

/// The order of a collection of items in bins, as a requirement or a
/// guarantee. The kinds are ordered by strength; two values are comparable
/// only where their keys agree.
struct OrderSpec {
  enum class Kind {
    /// No consumer: the bottom.
    None,
    /// Which items are present.
    Membership,
    /// The items of each group contiguous, at the offsets of `group`.
    GroupedBy,
    /// And ascending by `key` within a group; ties in any order.
    SortedWithin,
    /// And ties in the order of the items: every position fixed.
    Stable,
    /// Every element and every alias: what a reader by index sees. The top.
    Layout,
  };
  Kind kind = Kind::None;
  /// The root of the offsets (`start`) that define the groups.
  mlir::Value group;
  /// The root of the key buffer (`by`); null for the item itself.
  mlir::Value key;

  static OrderSpec none() { return {}; }
  static OrderSpec membership() { return {Kind::Membership, {}, {}}; }
  static OrderSpec groupedBy(mlir::Value group) {
    return {Kind::GroupedBy, group, {}};
  }
  /// `sorted_within(G, item)` is `stable(G, item)`: the item is a total
  /// order.
  static OrderSpec sortedWithin(mlir::Value group, mlir::Value key) {
    return {key ? Kind::SortedWithin : Kind::Stable, group, key};
  }
  static OrderSpec stable(mlir::Value group, mlir::Value key) {
    return {Kind::Stable, group, key};
  }
  static OrderSpec layout() { return {Kind::Layout, {}, {}}; }

  /// Whether `this` is met by `other` (`this` ⊑ `other`).
  bool isMetBy(const OrderSpec &other) const;
  /// The least value that meets both; `layout` where their keys differ.
  static OrderSpec join(const OrderSpec &a, const OrderSpec &b);
  bool operator==(const OrderSpec &other) const {
    return kind == other.kind && group == other.group && key == other.key;
  }
  void print(llvm::raw_ostream &os, mlir::AsmState &state) const;
};

/// How a consumer reads a collection.
enum class Access { Streamed, Gathered, Random };

/// The number of elements of a collection.
struct Cardinality {
  enum class Kind { Exact, Bound, RunTime } kind = Kind::RunTime;
  /// The index value it equals or is bounded by, plus `offset`; null where
  /// only the run-time size of the buffer is known.
  mlir::Value value;
  int64_t offset = 0;
  /// The constant, for `Exact`.
  int64_t exact = 0;
};

/// One use of a collection, by an op after its producer.
struct CollectionUse {
  enum class Role {
    /// Reads the collection: the requirement applies.
    Read,
    /// Writes it: an md_alg op's declared output (no requirement), or
    /// another op (`layout`).
    Write,
    /// Releases it (`gpu.dealloc`, `memref.dealloc`).
    Release,
    /// Reads its shape alone (`memref.dim`).
    Shape,
    /// It leaves what the analysis sees: `layout`.
    Escape,
  };
  mlir::Operation *op = nullptr;
  Role role = Role::Read;
  OrderSpec required;
  Access access = Access::Random;
  /// Whether its combination is exact (integers, slots of single items) or
  /// its determinism is `unordered`: what an elided sort needs as well.
  bool exactOrUnordered = false;
  /// The reason, for the printer.
  std::string why;
};

/// A buffer that an md_alg op writes, by the op and the operand's role.
struct Collection {
  mlir::Operation *producer = nullptr;
  std::string role;
  mlir::Value buffer;
  /// The allocation or the argument of the function that the buffer views.
  mlir::Value root;
  /// The views of the root that the analysis follows.
  llvm::SmallVector<mlir::Value> aliases;
  /// What the producer guarantees of its order.
  OrderSpec guaranteed;
  /// The join of what its consumers require.
  OrderSpec required;
  llvm::SmallVector<CollectionUse> uses;
  /// The worst access of a reader, if any.
  std::optional<Access> access;
  Cardinality cardinality;
  /// Why the buffer escapes, or empty.
  std::string escape;
  /// Positions in the pre-order walk of the function (the host's order of
  /// its launches): the allocation (-1 for an argument), the producer,
  /// the last use, and the release (-1 for none).
  int64_t allocated = -1, produced = -1, lastUse = -1, released = -1;
  /// The producer's determinism requirement, if it states one.
  std::optional<Determinism> determinism;
};

/// The verdict on a sort (`sort_by_key` with the order `stable`, or
/// `sort_bins`).
struct SortVerdict {
  mlir::Operation *sort = nullptr;
  /// The collection the sort produces.
  const Collection *result = nullptr;
  /// What the result would be without the sort.
  OrderSpec without;
  /// Whether the sort meets the requirement: its guarantee does, or the
  /// requirement is the layout, which the program reads with the sort.
  bool sufficient = false;
  /// Whether the requirement is met without the sort.
  bool orderMetWithout = false;
  /// Whether every consumer combines exactly or requires `unordered`.
  bool exactOrUnordered = false;
  bool redundant() const { return orderMetWithout && exactOrUnordered; }
};

/// The analysis of every function under an op (a module).
class CollectionAnalysis {
public:
  explicit CollectionAnalysis(mlir::Operation *root);

  llvm::ArrayRef<Collection> getCollections() const { return collections; }
  llvm::ArrayRef<SortVerdict> getSorts() const { return sorts; }
  /// The collection of `producer` in `role`, or null.
  const Collection *lookup(mlir::Operation *producer,
                           llvm::StringRef role) const;
  /// The facts, function by function, in the order of the walk.
  void print(llvm::raw_ostream &os) const;

private:
  mlir::Operation *root;
  llvm::SmallVector<Collection, 0> collections;
  llvm::SmallVector<SortVerdict, 0> sorts;
  llvm::DenseMap<mlir::Operation *, int64_t> position;
};

} // namespace mdir::md_alg

#endif // MDIR_DIALECT_MDALG_COLLECTIONANALYSIS_H
