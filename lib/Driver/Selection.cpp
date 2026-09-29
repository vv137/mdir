// Selections of particles by the names and numbers of their atoms and
// residues.

#include "mdir/Driver/Selection.h"

#include "mdir/Driver/Topology.h"

#include "llvm/ADT/StringExtras.h"

using namespace mdir::driver;
using llvm::StringRef;

namespace {
/// Whether `name` matches `pattern`, in which `*` is any run of characters
/// and `?` any one.
bool matches(StringRef pattern, StringRef name) {
  if (pattern.empty())
    return name.empty();
  if (pattern.front() == '*') {
    for (size_t skip = 0; skip <= name.size(); ++skip)
      if (matches(pattern.drop_front(), name.drop_front(skip)))
        return true;
    return false;
  }
  if (name.empty())
    return false;
  if (pattern.front() != '?' && pattern.front() != name.front())
    return false;
  return matches(pattern.drop_front(), name.drop_front());
}

/// A recursive descent over the text of a mask.
class Parser {
public:
  Parser(StringRef text, const Topology &topology)
      : text(text), topology(topology),
        count(topology.atomNames.size()) {}

  llvm::Expected<std::vector<bool>> parse() {
    auto result = parseOr();
    if (!result)
      return result.takeError();
    skipSpace();
    if (position != text.size())
      return fail("unexpected '" + text.substr(position, 1).str() + "'");
    return result;
  }

private:
  llvm::Error fail(const llvm::Twine &message) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(), "in the mask '%s', at %zu: %s",
        text.str().c_str(), position + 1, message.str().c_str());
  }

  void skipSpace() {
    while (position < text.size() && llvm::isSpace(text[position]))
      ++position;
  }
  bool accept(char c) {
    skipSpace();
    if (position < text.size() && text[position] == c) {
      ++position;
      return true;
    }
    return false;
  }

  llvm::Expected<std::vector<bool>> parseOr() {
    auto left = parseAnd();
    if (!left)
      return left.takeError();
    while (accept('|')) {
      auto right = parseAnd();
      if (!right)
        return right.takeError();
      for (size_t i = 0; i != count; ++i)
        (*left)[i] = (*left)[i] || (*right)[i];
    }
    return left;
  }

  llvm::Expected<std::vector<bool>> parseAnd() {
    auto left = parseUnary();
    if (!left)
      return left.takeError();
    while (accept('&')) {
      auto right = parseUnary();
      if (!right)
        return right.takeError();
      for (size_t i = 0; i != count; ++i)
        (*left)[i] = (*left)[i] && (*right)[i];
    }
    return left;
  }

  llvm::Expected<std::vector<bool>> parseUnary() {
    if (accept('!')) {
      auto inner = parseUnary();
      if (!inner)
        return inner.takeError();
      for (size_t i = 0; i != count; ++i)
        (*inner)[i] = !(*inner)[i];
      return inner;
    }
    if (accept('(')) {
      auto inner = parseOr();
      if (!inner)
        return inner.takeError();
      if (!accept(')'))
        return fail("expected ')'");
      return inner;
    }
    if (accept('*'))
      return std::vector<bool>(count, true);
    skipSpace();
    if (position < text.size() &&
        (text[position] == ':' || text[position] == '@'))
      return parsePrimary();
    return fail("expected ':', '@', '*', '!', or '('");
  }

  /// `:residues`, `@atoms`, or `:residues@atoms`.
  llvm::Expected<std::vector<bool>> parsePrimary() {
    std::vector<bool> result(count, true);
    if (accept(':')) {
      auto items = parseItems();
      if (!items)
        return items.takeError();
      for (size_t i = 0; i != count; ++i) {
        unsigned residue = topology.residueOf[i];
        result[i] = contains(*items, residue + 1,
                             topology.residueNames[residue]);
      }
    }
    if (accept('@')) {
      auto items = parseItems();
      if (!items)
        return items.takeError();
      for (size_t i = 0; i != count; ++i)
        result[i] = result[i] && contains(*items, i + 1, topology.atomNames[i]);
    }
    return result;
  }

  /// A number, a range of numbers, or a pattern of names.
  struct Item {
    bool isRange;
    size_t first, last;
    std::string pattern;
  };

  static bool contains(const std::vector<Item> &items, size_t number,
                       StringRef name) {
    for (const Item &item : items) {
      if (item.isRange ? number >= item.first && number <= item.last
                       : matches(item.pattern, name))
        return true;
    }
    return false;
  }

  /// A list of items separated by commas, up to an operator, a space, a
  /// parenthesis, or `@`.
  llvm::Expected<std::vector<Item>> parseItems() {
    std::vector<Item> items;
    do {
      skipSpace();
      size_t begin = position;
      while (position < text.size() &&
             !llvm::StringRef(",&|!()@: \t").contains(text[position]))
        ++position;
      StringRef token = text.slice(begin, position);
      if (token.empty())
        return fail("expected a number, a range, or a name");
      if (llvm::isDigit(token.front())) {
        auto [low, high] = token.split('-');
        bool isRange = token.contains('-');
        size_t first = 0, last = 0;
        bool bad = low.getAsInteger(10, first);
        if (isRange)
          bad = bad || high.getAsInteger(10, last);
        else
          last = first;
        if (bad || first == 0 || last < first)
          return fail("expected a number or a range such as 1-10, not '" +
                      token.str() + "'");
        items.push_back({true, first, last, ""});
      } else {
        items.push_back({false, 0, 0, token.str()});
      }
    } while (position < text.size() && text[position] == ',' && ++position);
    return items;
  }

  StringRef text;
  const Topology &topology;
  size_t count;
  size_t position = 0;
};
} // namespace

llvm::Expected<std::vector<bool>>
mdir::driver::selectParticles(StringRef text, const Topology &topology) {
  return Parser(text, topology).parse();
}
