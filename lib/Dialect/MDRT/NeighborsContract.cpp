// The layout contract of a neighbor structure; see
// include/mdir/Dialect/MDRT/NeighborsContract.h.

#include "mdir/Dialect/MDRT/NeighborsContract.h"

using namespace mdir::mdrt;

using Contract = NeighborsContract;

llvm::StringRef mdir::mdrt::stringify(Contract::Traversal value) {
  return value == Contract::Traversal::Directed ? "directed" : "unique";
}

llvm::StringRef mdir::mdrt::stringify(Contract::Images value) {
  return value == Contract::Images::Minimum ? "minimum" : "explicit";
}

llvm::StringRef mdir::mdrt::stringify(Contract::Exclusions value) {
  return value == Contract::Exclusions::Masked ? "masked" : "removed";
}

llvm::StringRef mdir::mdrt::stringify(Contract::Group value) {
  switch (value) {
  case Contract::Group::None:
    return "";
  case Contract::Group::Owner:
    return "owner";
  case Contract::Group::Group:
    return "group";
  }
  return "";
}

llvm::StringRef mdir::mdrt::stringify(Contract::Order value) {
  switch (value) {
  case Contract::Order::None:
    return "";
  case Contract::Order::Item:
    return "item";
  case Contract::Order::Search:
    return "search";
  }
  return "";
}

Contract Contract::matrix(Images images) {
  Contract contract;
  contract.traversal = Traversal::Directed;
  contract.images = images;
  contract.exclusions = Exclusions::Masked;
  contract.organization = Organization::Stable;
  contract.group = Group::Owner;
  contract.order = Order::Search;
  return contract;
}

Contract Contract::groups() {
  Contract contract;
  contract.traversal = Traversal::Unique;
  contract.images = Images::Minimum;
  contract.exclusions = Exclusions::Removed;
  contract.organization = Organization::Stable;
  contract.group = Group::Group;
  contract.order = Order::Search;
  return contract;
}

/// Returns true if a sort by `order` leaves no ties: the item and the
/// order of the search are total.
static bool isTotal(Contract::Order order) {
  return order == Contract::Order::Item || order == Contract::Order::Search;
}

Contract Contract::withOrganization(Contract base, Organization organization,
                                    Group group, Order order) {
  if (organization == Organization::SortedWithin && isTotal(order))
    organization = Organization::Stable;
  base.organization = organization;
  base.group = organization == Organization::None ? Group::None : group;
  base.order = organization == Organization::None ||
                       organization == Organization::GroupedBy
                   ? Order::None
                   : order;
  return base;
}

bool Contract::isNormalized() const {
  switch (organization) {
  case Organization::None:
    return group == Group::None && order == Order::None;
  case Organization::GroupedBy:
    return group != Group::None && order == Order::None;
  case Organization::SortedWithin:
    return group != Group::None && order != Order::None && !isTotal(order);
  case Organization::Stable:
    return group != Group::None && order != Order::None;
  }
  return false;
}

/// Returns true if the organization `guaranteed` meets `required`: the
/// order of D409, in which two values with keys are comparable only if
/// their keys agree.
static bool meetsOrganization(const Contract &guaranteed,
                              const Contract &required) {
  using Organization = Contract::Organization;
  if (required.organization == Organization::None)
    return true;
  if (guaranteed.organization < required.organization)
    return false;
  if (guaranteed.group != required.group)
    return false;
  if (required.organization == Organization::GroupedBy)
    return true;
  return guaranteed.order == required.order;
}

bool Contract::meets(const Contract &required) const {
  return !whyNot(required);
}

std::optional<std::string> Contract::whyNot(const Contract &required) const {
  std::string reason;
  llvm::raw_string_ostream os(reason);
  if (traversal != required.traversal) {
    os << "the traversal '" << stringify(required.traversal)
       << "' is required, and the structure is '" << stringify(traversal)
       << "'";
    return reason;
  }
  if (images != required.images) {
    os << "the images '" << stringify(required.images)
       << "' are required, and the structure's are '" << stringify(images)
       << "'";
    return reason;
  }
  if (exclusions < required.exclusions) {
    os << "the exclusions '" << stringify(required.exclusions)
       << "' are required, and the structure's are '"
       << stringify(exclusions) << "'";
    return reason;
  }
  if (!meetsOrganization(*this, required)) {
    os << "the organization '";
    required.printOrganization(os);
    os << "' is required, and the structure guarantees '";
    printOrganization(os);
    os << "'";
    return reason;
  }
  return std::nullopt;
}

std::optional<Contract> Contract::joinOrganization(const Contract &a,
                                                   const Contract &b) {
  if (meetsOrganization(a, b))
    return a;
  if (meetsOrganization(b, a))
    return b;
  return std::nullopt;
}

void Contract::printOrganization(llvm::raw_ostream &os) const {
  switch (organization) {
  case Organization::None:
    os << "none";
    return;
  case Organization::GroupedBy:
    os << "grouped_by(" << stringify(group) << ")";
    return;
  case Organization::SortedWithin:
    os << "sorted_within(" << stringify(group) << ", " << stringify(order)
       << ")";
    return;
  case Organization::Stable:
    os << "stable(" << stringify(group) << ", " << stringify(order) << ")";
    return;
  }
}

void Contract::print(llvm::raw_ostream &os) const {
  os << stringify(traversal) << ", " << stringify(images) << ", "
     << stringify(exclusions) << ", ";
  printOrganization(os);
}

std::string Contract::str() const {
  std::string text;
  llvm::raw_string_ostream os(text);
  print(os);
  return text;
}
