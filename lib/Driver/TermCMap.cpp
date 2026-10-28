// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace mdir::driver::builder {
class CMapTermBuilder final : public TermBuilder {
public:
  CMapTermBuilder() { tupleRank = 2; }
  int topologyRank() const override { return 4; }
  void collectTupleSets(TermParameters &context) const override {
    const Topology &topology = context.getTopology();
    if (!topology.cmaps.empty()) {
      // The map of each term, and the bicubic patches of every map: a row
      // of 16 coefficients for each cell.
      Program::TupleSet &set = addTupleSet(context.getTupleSets(), "cmap", 5);
      set.reversible = false;
      size_t map = addTupleField(set, "map");
      for (const Topology::CMap &cmap : topology.cmaps) {
        for (unsigned member : {cmap.i, cmap.j, cmap.k, cmap.l, cmap.m})
          set.members.push_back(member);
        set.fields[map].values.push_back(cmap.map);
      }
      Program::Table table;
      table.name = "cmap";
      table.values = getCMapCoefficients(topology);
      table.columns = 16;
      table.count = table.values.size() / 16;
      context.getTables().push_back(std::move(table));
    }
  }
  void appendTermDefinitions(
      const TermInputs &,
      std::vector<TermDefinition> &definitions) const override {
    definitions.push_back({7, 0, "term_cmap", TermInputs::CMaps});
  }
  void emitTopologyPotential(TermEmission &context,
                             const PotentialSelection &selection,
                             const TermPotentialAddition &add) const override {
    auto &os = context.getOutput();
    unsigned terms = selection.terms;
    auto has = [&](StringRef name) {
      return llvm::any_of(context.getTupleSets(),
                          [&](const auto &set) { return set.name == name; });
    };
    if ((terms & TermInputs::CMaps) && has("cmap")) {
      // The cell of φ and ψ, each from −180° in steps of 360° / n, and the
      // places t and u in it; φ = 180° is the first cell again. The patch
      // is E = Σ c_ij t^i u^j [MacKerell2004].
      unsigned n = context.getTopology().cmapResolution;
      os << "  %u_cmap = md.sum_tuples %r_cmap, %x, %cell coordinates("
            "dihedral(0, 1, 2, 3), dihedral(1, 2, 3, 4))\n"
         << "      tuple(%f_cmap_map : !of_cmap) {\n"
         << "  ^bb0(%phi: f64, %psi: f64, %map: f64):\n"
         << "    %scale = arith.constant " << formatReal(n / (2.0 * M_PI))
         << " : f64\n"
         << "    %shift = arith.constant " << formatReal(n / 2.0) << " : f64\n"
         << "    %cmap_n = arith.constant " << n << " : i32\n";
      for (StringRef angle : {"phi", "psi"}) {
        std::string a = angle.str();
        os << "    %" << a << "_s = arith.mulf %" << a << ", %scale : f64\n"
           << "    %" << a << "_x = arith.addf %" << a << "_s, %shift : f64\n"
           << "    %" << a << "_f = math.floor %" << a << "_x : f64\n"
           << "    %" << a << "_t = arith.subf %" << a << "_x, %" << a
           << "_f : f64\n"
           << "    %" << a << "_i = arith.fptosi %" << a << "_f : f64 to i32\n"
           << "    %" << a << "_j = arith.addi %" << a << "_i, %cmap_n : i32\n"
           << "    %" << a << "_k = arith.remsi %" << a
           << "_j, %cmap_n : i32\n";
      }
      os << "    %m = arith.fptosi %map : f64 to i32\n"
         << "    %mn = arith.muli %m, %cmap_n : i32\n"
         << "    %row = arith.addi %mn, %phi_k : i32\n"
         << "    %rown = arith.muli %row, %cmap_n : i32\n"
         << "    %cellid = arith.addi %rown, %psi_k : i32\n";
      for (int k = 0; k != 16; ++k)
        os << "    %k" << k << " = arith.constant " << k << " : i32\n"
           << "    %c" << k << " = md.lookup %t_cmap[%cellid, %k" << k
           << "] : !grid, i32, i32 -> f64\n";
      // Horner in u for each power of t, then in t.
      for (int i = 0; i != 4; ++i) {
        std::string last = "%c" + std::to_string(4 * i + 3);
        for (int j = 2; j >= 0; --j) {
          std::string name = "%r" + std::to_string(i) + std::to_string(j);
          os << "    " << name << "m = arith.mulf " << last
             << ", %psi_t : f64\n"
             << "    " << name << " = arith.addf " << name << "m, %c"
             << 4 * i + j << " : f64\n";
          last = name;
        }
      }
      std::string last = "%r30";
      for (int i = 2; i >= 0; --i) {
        std::string name = "%e" + std::to_string(i);
        os << "    " << name << "m = arith.mulf " << last << ", %phi_t : f64\n"
           << "    " << name << " = arith.addf " << name << "m, %r" << i
           << "0 : f64\n";
        last = name;
      }
      os << "    md.yield %e0 : f64\n"
         << "  } : !rel_cmap, !vec -> f64\n";
      add("cmap");
    }
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder> mdir::driver::builder::createCMapTermBuilder() {
  return std::make_unique<CMapTermBuilder>();
}
