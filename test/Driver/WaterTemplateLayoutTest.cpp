// Check physical layout boundaries without allocating device memory.
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mdir/Runtime/TemplateMembership.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cstdlib>

static void requireLayout(bool condition, const char *message) {
  if (!condition) {
    llvm::errs() << message << '\n';
    std::exit(1);
  }
}

int runWaterTemplateLayoutTest() {
  unsigned checked = 0;
  for (int64_t waters : {0, 1, 15, 16, 17, 1024})
    for (int64_t atoms : {0, 1, 15, 16, 17, 256})
      for (int64_t entries : {0, 1, 63, 64, 65, 4096}) {
        MdrtTemplatePairLayout layout;
        requireLayout(mdrtTemplatePairLayout(waters, atoms, entries, &layout),
                      "valid template layout refused");
        // Count actual clusters by walking units, independently of the
        // allocator's rounding expressions. SW and SS each have atom rows.
        int64_t rows = 0;
        for (int64_t i = 0; i < waters; i += 16) ++rows;
        for (int64_t i = 0; i < atoms; i += 16) rows += 2;
        requireLayout(layout.rows == rows, "partial class cluster lost");
        struct View { int64_t begin, count; };
        std::array<View, 7> views = {{{0, 48},
            {layout.outerRows, rows * 4}, {layout.innerRows, rows * 4},
            {layout.outerUnits, layout.units * 4},
            {layout.innerUnits, layout.units * 4},
            {layout.outerEntries, entries * 4},
            {layout.innerEntries, entries * 4}}};
        int64_t end = 0;
        for (View view : views) {
          requireLayout(view.begin == end && view.begin % 4 == 0,
                        "template views overlap, gap or lose alignment");
          end += view.count;
        }
        requireLayout(end == layout.words, "allocation omits a view's tail");
        // Worst fragmentation puts one partial block in each row before
        // filling any row. Check that packing the remaining entries into
        // the first row cannot exceed the work-unit capacity.
        if (rows) {
          int64_t remaining = entries;
          int64_t work = 0;
          for (int64_t row = 0; row < rows && remaining; ++row) {
            --remaining;
            ++work;
          }
          if (remaining) work += remaining / 64 + (remaining % 64 != 0);
          requireLayout(work <= layout.units, "partial work units overflow");
        }
        ++checked;
      }
  MdrtTemplatePairLayout layout;
  requireLayout(!mdrtTemplatePairLayout(-1, 0, 0, &layout) &&
                !mdrtTemplatePairLayout(0, -1, 0, &layout) &&
                !mdrtTemplatePairLayout(0, 0, -1, &layout) &&
                !mdrtTemplatePairLayout(INT32_MAX, 1, 0, &layout) &&
                !mdrtTemplatePairLayout(0, 0, int64_t(INT32_MAX) + 1, &layout) &&
                !mdrtTemplatePairLayout(0, 0, 0, nullptr),
                "invalid template index domain admitted");
  requireLayout(mdrtTemplatePairLayout(INT32_MAX, 0, INT32_MAX, &layout) &&
                layout.words > INT32_MAX && layout.innerEntries > INT32_MAX,
                "large template physical offsets narrowed to i32");
  // Renumbered OHHM tuples: EPs 0 and 3 must not reappear as solute.
  int32_t members[] = {4, 2, 7, 0, 6, 1, 8, 3};
  int32_t units[10] = {};
  requireLayout(mdrtTemplateMembership(10, 2, 4, members, 4, 1, units, 8),
                "renumbered four-site membership refused");
  constexpr int32_t expected[] = {4, 2, 7, 6, 1, 8, 5, 9};
  for (unsigned i = 0; i != 8; ++i)
    requireLayout(units[i] == expected[i], "extra point leaked into atom class");
  // Three-site input with interleaved columns tests the descriptor strides.
  int32_t strided[] = {4, -1, 2, -1, 7, -1};
  requireLayout(mdrtTemplateMembership(8, 1, 3, strided, 6, 2, units, 8) &&
                units[0] == 4 && units[1] == 2 && units[2] == 7 &&
                units[3] == 0 && units[7] == 6,
                "strided membership or atom complement incorrect");
  members[7] = 4;
  requireLayout(!mdrtTemplateMembership(10, 2, 4, members, 4, 1, units, 8) &&
                !mdrtTemplateMembership(8, 1, 3, strided, 6, 2, units, 7),
                "overlapping membership or short destination admitted");
  llvm::outs() << "template allocation: " << checked
               << " layouts; invalid domains refused; wide offsets preserved\n"
               << "template membership: real units, EP exclusion and strides passed\n";
  return 0;
}
