// The independent, image-aware displacements of a template window.
#ifndef MDIR_DIALECT_MD_MDTEMPLATEWINDOW_H
#define MDIR_DIALECT_MD_MDTEMPLATEWINDOW_H
#include "llvm/ADT/SmallVector.h"
#include <array>
namespace mdir::md {
/// (positive member, negative member), with the anchor image in the first
/// coordinate. Remaining coordinates are whole-molecule H-O offsets.
llvm::SmallVector<std::array<unsigned, 2>, 5>
getTemplateWindowMembers(unsigned arity);
}
#endif
