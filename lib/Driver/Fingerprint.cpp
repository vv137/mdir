// What defined a run (D172).

#include "mdir/Driver/Fingerprint.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <tuple>

#define TOML_EXCEPTIONS 0
#define TOML_ENABLE_FORMATTERS 0
#include "toml.hpp"

using namespace mdir::driver;
using llvm::StringRef;

namespace {

/// A value longer than this is recorded by its hash, so that a checkpoint
/// stays small and a message stays readable.
constexpr size_t longestShown = 160;

std::string hashBytes(StringRef data) {
  return "sha256:" +
         llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(data)),
                     /*LowerCase=*/true);
}

/// A number as the shortest text that reads back to the same double, so
/// that `9` and `9.0` are the same value.
std::string formatNumber(double value) {
  char buffer[32];
  if (value == std::trunc(value) && std::fabs(value) < 1e15) {
    std::snprintf(buffer, sizeof(buffer), "%.0f", value);
    return buffer;
  }
  for (int digits = 1; digits <= 17; ++digits) {
    std::snprintf(buffer, sizeof(buffer), "%.*g", digits, value);
    if (std::strtod(buffer, nullptr) == value)
      break;
  }
  return buffer;
}

/// An integer as its exact decimal text (#620). Below 2^53 in magnitude
/// an integer is a double exactly, and its text is that of the double, as
/// fingerprints wrote every integer before: the checkpoints of such values
/// continue. From 2^53 on, the double rounds and gave distinct integers one
/// text (2^53 and 2^53 + 1, 2^63 - 2 and 2^63 - 1); those are now written
/// with all their digits.
std::string formatInteger(bool negative, uint64_t magnitude) {
  if (magnitude < (uint64_t(1) << 53)) {
    double value = static_cast<double>(magnitude);
    return formatNumber(negative ? -value : value);
  }
  return (negative ? "-" : "") + std::to_string(magnitude);
}

std::string quote(StringRef text) {
  std::string result = "\"";
  for (char c : text) {
    if (c == '"' || c == '\\')
      result += '\\';
    if (c == '\n') {
      result += "\\n";
      continue;
    }
    result += c;
  }
  return result + "\"";
}

/// The spellings of the choices of the control file, by the place where
/// each value begins (Control::choiceSpellings).
using Spellings = std::map<std::pair<uint32_t, uint32_t>, std::string>;

/// The canonical text of a value of a control file: the keys of a table in
/// the order of their names, a floating-point number as the shortest text of
/// its double, an integer as its exact decimal (formatInteger), a string
/// that names a choice in the spelling that the reader documents for it
/// (#702) and any other string as written, and no spaces.
std::string canonical(const toml::node &node, const Spellings &spellings) {
  if (const toml::table *table = node.as_table()) {
    std::map<std::string, const toml::node *> sorted;
    for (const auto &[key, value] : *table)
      sorted[std::string(key.str())] = &value;
    std::string result = "{";
    bool first = true;
    for (const auto &[key, value] : sorted) {
      if (!first)
        result += ",";
      first = false;
      result += key + "=" + canonical(*value, spellings);
    }
    return result + "}";
  }
  if (const toml::array *array = node.as_array()) {
    std::string result = "[";
    for (size_t i = 0, e = array->size(); i != e; ++i) {
      if (i != 0)
        result += ",";
      result += canonical(*array->get(i), spellings);
    }
    return result + "]";
  }
  if (const auto *text = node.as_string()) {
    // The reader read this file and recorded its choices; this is the same
    // file, read again, so a value begins where it did then.
    const toml::source_position &begin = node.source().begin;
    auto found = spellings.find({begin.line, begin.column});
    if (found != spellings.end())
      return quote(found->second);
    return quote(text->get());
  }
  if (const auto *integer = node.as_integer())
    return getFingerprintInteger(integer->get());
  if (const auto *real = node.as_floating_point())
    return formatNumber(real->get());
  if (const auto *flag = node.as_boolean())
    return flag->get() ? "true" : "false";
  // Dates and times have no meaning in a control file; their type is
  // enough to tell that one changed.
  return "<type " + std::to_string(static_cast<int>(node.type())) + ">";
}

std::string shown(std::string text) {
  if (text.size() > longestShown)
    return hashBytes(text);
  return text;
}

} // namespace

std::string mdir::driver::getFingerprintNumber(double value) {
  return formatNumber(value);
}

std::string mdir::driver::getFingerprintInteger(int64_t value) {
  // The magnitude of INT64_MIN is not an int64_t.
  uint64_t magnitude = value < 0 ? uint64_t(-(value + 1)) + 1 : uint64_t(value);
  return formatInteger(value < 0, magnitude);
}

std::string mdir::driver::getFingerprintUnsigned(uint64_t value) {
  return formatInteger(false, value);
}

std::string mdir::driver::getFingerprintString(StringRef text) {
  return quote(text);
}

std::string mdir::driver::getFingerprintHash(StringRef data) {
  return hashBytes(data);
}

std::string
mdir::driver::getTopologyFilesHash(llvm::ArrayRef<std::string> contents) {
  std::string hashes;
  for (const std::string &content : contents)
    hashes += hashBytes(content) + "\n";
  return hashBytes(hashes);
}

std::string mdir::driver::getNumbersHash(const std::vector<double> &numbers) {
  return hashBytes(StringRef(reinterpret_cast<const char *>(numbers.data()),
                             numbers.size() * sizeof(double)));
}

llvm::Expected<Fingerprint>
mdir::driver::getRunFingerprint(StringRef controlFile, const Control &control,
                                const System &system) {
  toml::parse_result parsed = toml::parse_file(std::string_view(controlFile));
  if (!parsed)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot read '%s' again",
                                   controlFile.str().c_str());
  toml::table &root = parsed.table();

  // Compare the grid actually loaded, rather than its filename. Reconstruct
  // the inline representation so that a relocated file or equivalent inline
  // values define the same physics (D178).
  if (toml::array *functions = root["energy"]["function"].as_array())
    for (toml::node &node : *functions) {
      toml::table *table = node.as_table();
      if (!table || !table->contains("values_file"))
        continue;
      auto name = (*table)["name"].value<std::string>();
      auto found = llvm::find_if(control.functions, [&](const auto &function) {
        return name && function.name == *name;
      });
      if (found == control.functions.end())
        continue;
      std::vector<size_t> stride(found->sizes.size(), 1);
      for (size_t k = 1; k < stride.size(); ++k)
        stride[k] = stride[k - 1] * found->sizes[k - 1];
      std::function<toml::array(unsigned, size_t)> values =
          [&](unsigned depth, size_t offset) {
        toml::array result;
        for (unsigned i = 0; i < found->sizes[depth]; ++i) {
          size_t at = offset + i * stride[depth];
          if (depth + 1 == found->sizes.size())
            result.push_back(found->values[at]);
          else
            result.push_back(values(depth + 1, at));
        }
        return result;
      };
      table->erase("values_file");
      table->erase("shape");
      table->insert("values", values(0, 0));
    }

  Fingerprint fingerprint;
  auto add = [&](StringRef group, std::string name, std::string value,
                 std::string written = {}) {
    if (written == value)
      written.clear();
    fingerprint.push_back(
        {group.str(), std::move(name), std::move(value), std::move(written)});
  };
  // The keys of the tables of a group, each an entry; a table that is an
  // array, such as [[restraints]], is one.
  auto addTables = [&](StringRef group,
                       std::initializer_list<StringRef> tables,
                       std::set<std::string> skipped = {}) {
    for (StringRef name : tables) {
      const toml::node *node = root.get(std::string_view(name));
      if (!node)
        continue;
      if (const toml::table *table = node->as_table()) {
        for (const auto &[key, value] : *table) {
          std::string keyName(key.str());
          if (skipped.count(keyName))
            continue;
          // An array of tables is named as the file writes its entries.
          const toml::array *array = value.as_array();
          bool tables = array && !array->empty() &&
                        array->is_array_of_tables();
          // The reach and the rebuilds of the neighbor structures decide how
          // the forces are found, not what they are: they may change as
          // [execution] may.
          bool neighbors =
              name == "energy" &&
              (keyName == "pairlist_distance" ||
               keyName == "pruned_distance" || keyName == "rebuild_interval");
          add(neighbors ? StringRef("execution") : group,
              tables ? "[[" + name.str() + "." + keyName + "]]"
                     : "[" + name.str() + "] " + keyName,
              shown(canonical(value, control.choiceSpellings)),
              shown(canonical(value, {})));
        }
        continue;
      }
      add(group, "[[" + name.str() + "]]",
          shown(canonical(*node, control.choiceSpellings)),
          shown(canonical(*node, {})));
    }
  };
  addTables("physics", {"energy", "pme", "lj_pme", "constraints",
                        "restraints", "boundary", "free_energy"});
  addTables("coupling", {"dynamics", "ensemble", "thermostat", "barostat"},
            {"steps"});
  addTables("execution", {"execution"});

  // The files that define the topology, by their contents: another name
  // for the same file is not a change.
  std::vector<std::string> files;
  for (const std::string *path :
       {&control.prmtopFile, &control.gromacsTopologyFile,
        &control.charmmStructureFile})
    if (!path->empty())
      files.push_back(*path);
  for (const std::string &path : control.charmmParameterFiles)
    files.push_back(path);
  if (system.topology)
    for (const std::string &path : system.topology->sourceFiles)
      if (std::find(files.begin(), files.end(), path) == files.end())
        files.push_back(path);
  if (!files.empty()) {
    std::vector<std::string> contents;
    for (const std::string &path : files) {
      auto buffer = llvm::MemoryBuffer::getFile(path);
      if (!buffer)
        return llvm::createStringError(buffer.getError(), "cannot read '%s'",
                                       path.c_str());
      contents.push_back((*buffer)->getBuffer().str());
    }
    add("physics", "the files of the topology",
        getTopologyFilesHash(contents));
  }

  add("physics", "the masses", getNumbersHash(system.masses));
  if (!control.restraints.empty())
    add("physics", "the reference of the restraints",
        getNumbersHash(system.referencePositions));

  std::stable_sort(fingerprint.begin(), fingerprint.end(),
                   [](const FingerprintEntry &a, const FingerprintEntry &b) {
                     return std::tie(a.group, a.name) <
                            std::tie(b.group, b.name);
                   });
  return fingerprint;
}

std::vector<FingerprintChange>
mdir::driver::compareFingerprints(const Fingerprint &stored,
                                  const Fingerprint &current,
                                  StringRef group) {
  std::map<std::string, std::pair<std::string, std::string>> values;
  for (const FingerprintEntry &entry : stored)
    if (entry.group == group)
      values[entry.name] = {entry.value, "(none)"};
  for (const FingerprintEntry &entry : current)
    if (entry.group == group) {
      auto [found, inserted] =
          values.try_emplace(entry.name, "(none)", entry.value);
      if (inserted)
        continue;
      // A checkpoint written before #702 holds a choice in the case of its
      // file: the same file continues it as it did then.
      found->second.second =
          !entry.written.empty() && found->second.first == entry.written
              ? entry.written
              : entry.value;
    }
  std::vector<FingerprintChange> changes;
  for (const auto &[name, pair] : values)
    if (pair.first != pair.second)
      changes.push_back({group.str(), name, pair.first, pair.second});
  return changes;
}
