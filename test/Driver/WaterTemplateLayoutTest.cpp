// Check physical layout boundaries without allocating device memory.
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mdir/Runtime/TemplateMembership.h"
#include "mdir/Runtime/TemplateExclusions.h"
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
        std::array<View, 14> views = {{{0, 48},
            {layout.outerRows, rows * 4}, {layout.innerRows, rows * 4},
            {layout.outerUnits, layout.units * 4},
            {layout.innerUnits, layout.units * 4},
            {layout.outerEntries, entries * 4},
            {layout.innerEntries, entries * 4},
            {layout.stageUnits, layout.units * 4},
            {layout.stageEntries, layout.units * 64 * 4},
            {layout.buildCounts, rows},
            {layout.buildEntryOffsets, rows + 1},
            {layout.buildWorkCounts, rows},
            {layout.buildWorkOffsets, rows + 1},
            {layout.buildScanScratch, layout.scanBlocks}}};
        int64_t end = 0;
        for (View view : views) {
          requireLayout(view.begin == end,
                        "template views overlap, gap or lose alignment");
          end += view.count;
        }
        requireLayout(end == layout.words, "allocation omits a view's tail");
        requireLayout(layout.stageUnits % 4 == 0 && layout.stageEntries % 4 == 0 &&
                      layout.scanBlocks * 1024 >= rows && layout.scanBlocks >= 1,
                      "record alignment or row-scan capacity differs");
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
  requireLayout(layout.units * 64 > INT32_MAX &&
                layout.buildCounts > layout.stageEntries + INT32_MAX,
                "staging addresses narrowed to payload indices");
  requireLayout(mdrtTemplatePairLayout(1, int64_t(INT32_MAX) - 1, 0, &layout),
                "valid split class index domain refused");
  uint64_t lastAtomPlace = 16 + ((layout.atoms - 1) / 16) * 16;
  requireLayout(lastAtomPlace > INT32_MAX &&
                uint64_t(uint32_t(lastAtomPlace)) == lastAtomPlace,
                "padded class places lost their unsigned i32 index");
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
  MdrtTemplateExclusions graph{};
  const int32_t exclusions[] = {4, 2, 5, 9, 9, 5, 6, 8, 5, 5};
  requireLayout(mdrtTemplateExclusions(10, 2, 4, expected, 5, exclusions, 2, 1, &graph) &&
                graph.atoms == 2 && graph.count == 2 &&
                graph.rows[0] == 0 && graph.rows[1] == 1 && graph.rows[2] == 2 &&
                graph.neighbors[0] == 1 && graph.neighbors[1] == 0,
                "SS graph retained water edges or duplicate directions");
  mdrtFreeTemplateExclusions(&graph);
  const int32_t atoms[] = {0, 1, 2, 3, 4, 5, 6, 7};
  const int32_t stridedEdges[] = {7, 1, -1, 2, 5, -1, 1, 0, -1,
                                 1, 7, -1, 5, 2, -1, 4, 4, -1};
  requireLayout(mdrtTemplateExclusions(8, 0, 3, atoms, 6, stridedEdges, 3, 1, &graph),
                "strided SS graph refused");
  constexpr int32_t expectedRows[] = {0, 1, 3, 4, 4, 4, 5, 5, 6};
  constexpr int32_t expectedNeighbors[] = {1, 0, 7, 5, 2, 1};
  requireLayout(graph.count == 6, "SS graph has the wrong number of neighbors");
  for (unsigned i = 0; i != 9; ++i)
    requireLayout(graph.rows[i] == expectedRows[i], "SS CSR row boundary differs");
  for (unsigned i = 0; i != 6; ++i)
    requireLayout(graph.neighbors[i] == expectedNeighbors[i], "SS neighbors are not sorted");
  mdrtFreeTemplateExclusions(&graph);
  const int32_t badEdge[] = {0, 8};
  requireLayout(!mdrtTemplateExclusions(8, 0, 3, atoms, 1, badEdge, 2, 1, &graph),
                "out-of-domain SS endpoint admitted");
  requireLayout(mdrtTemplateExclusions(0, 0, 3, nullptr, 0, nullptr, 0, 0, &graph) &&
                graph.atoms == 0 && graph.count == 0 && graph.rows[0] == 0,
                "empty SS graph failed");
  mdrtFreeTemplateExclusions(&graph);
  llvm::outs() << "template allocation: " << checked
               << " layouts; invalid domains refused; wide offsets preserved\n"
               << "template membership: real units, EP exclusion and strides passed\n";
  llvm::outs() << "template exclusions: SS CSR, deduplication and empty domains passed\n";
  return 0;
}
