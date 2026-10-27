// Shared D355 effect summaries and singleton resources.
#include "mdir/Effects/Effects.h"

using namespace mlir;
using namespace mdir::dyn;

BathResource *mdir::dyn::BathResource::get() {
  static BathResource resource;
  return &resource;
}

ChainResource *mdir::dyn::ChainResource::get() {
  static ChainResource resource;
  return &resource;
}

StopResource *mdir::dyn::StopResource::get() {
  static StopResource resource;
  return &resource;
}

RecordResource *mdir::dyn::RecordResource::get() {
  static RecordResource resource;
  return &resource;
}

StatisticsResource *mdir::dyn::StatisticsResource::get() {
  static StatisticsResource resource;
  return &resource;
}

/// The resource of the name `name` in a summary of effects, or null.
static SideEffects::Resource *getResource(StringRef name) {
  for (SideEffects::Resource *resource :
       {static_cast<SideEffects::Resource *>(BathResource::get()),
        static_cast<SideEffects::Resource *>(ChainResource::get()),
        static_cast<SideEffects::Resource *>(StopResource::get()),
        static_cast<SideEffects::Resource *>(RecordResource::get()),
        static_cast<SideEffects::Resource *>(StatisticsResource::get()),
        static_cast<SideEffects::Resource *>(
            SideEffects::DefaultResource::get())})
    if (resource->getName() == name)
      return resource;
  return nullptr;
}

/// The effects that `summary`, `{read = [...], write = [...]}`, states, or
/// failure with a message in `error`.
LogicalResult mdir::effects::parseEffectSummary(
    DictionaryAttr summary,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
    std::string &error) {
  for (NamedAttribute entry : summary) {
    StringRef kind = entry.getName().getValue();
    if (kind != "read" && kind != "write") {
      error = ("unknown kind of effect '" + kind +
               "'; expected 'read' or 'write'")
                  .str();
      return failure();
    }
    auto names = dyn_cast<ArrayAttr>(entry.getValue());
    if (!names) {
      error = ("expected an array of the names of resources in '" + kind +
               "'")
                  .str();
      return failure();
    }
    for (Attribute name : names) {
      auto text = dyn_cast<StringAttr>(name);
      SideEffects::Resource *resource =
          text ? getResource(text.getValue()) : nullptr;
      if (!resource) {
        error = "unknown resource in '" + kind.str() + "'; known: dyn.bath, "
                "dyn.chain, dyn.stop, dyn.record, dyn.statistics, <Default>";
        return failure();
      }
      if (kind == "read")
        effects.emplace_back(MemoryEffects::Read::get(), resource);
      else
        effects.emplace_back(MemoryEffects::Write::get(), resource);
    }
  }
  return success();
}

