// D372: updates preserve the force field outside coefficients.
#include "mdir/Driver/Model.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

using namespace mdir;

static void require(bool ok, const char *message) {
  if (!ok) {
    llvm::errs() << message << '\n';
    std::exit(1);
  }
}
template <class T> static T take(llvm::Expected<T> value) {
  if (!value) {
    llvm::errs() << llvm::toString(value.takeError()) << '\n';
    std::exit(1);
  }
  return std::move(*value);
}
static void check(llvm::Error error) {
  if (error) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    std::exit(1);
  }
}

int main(int argc, char **argv) {
  require(argc == 3, "expected Amber topology and coordinates");
  auto loaded = take(model::loadAmber(argv[1], argv[2]));
  auto system = loaded.makeSystem();
  system.cutoff = 0.8;
  system.pairlistDistance = 0.9;
  system.dispersion = driver::DispersionCorrection::None;
  system.electrostatics = model::Electrostatics::Cutoff;
  model::Tunable declaration;
  declaration.name = "torsion_k";
  declaration.parameter = "dihedral_k";
  system.tunables = {declaration};
  system.tunableGradient = true;
  model::Execution execution;
  execution.target = driver::Target::CPU;
  execution.deterministic = true;
  auto prepared = take(model::prepare(system, loaded.makeState(), {}, {}, execution, {}));
  auto before = take(prepared.build());
  const auto original = prepared.system.topology;
  require(!original->pairs.empty(), "fixture needs nonempty 1-4 pairs");
  auto values = prepared.tunables.values;
  require(!values.empty() && !values[0].empty(), "fixture needs proper coefficients");
  for (size_t i = 0; i != values[0].size(); ++i)
    values[0][i] = i % 2 ? 0.0 : -0.5 * values[0][i];
  auto updated = prepared.system;
  check(model::applyTunables(prepared.tunables, values, prepared.control, updated));
  const auto &changed = *updated.topology;
  require(changed.exclusions == original->exclusions &&
          changed.sigma == original->sigma && changed.epsilon == original->epsilon &&
          changed.charges == original->charges, "update changed nonbonded arrays");
  require(changed.pairs.size() == original->pairs.size(), "update changed 1-4 count");
  for (auto [a, b] : llvm::zip_equal(changed.pairs, original->pairs))
    require(a.i == b.i && a.j == b.j && a.scaleLJ == b.scaleLJ &&
            a.scaleCoulomb == b.scaleCoulomb && a.sigma == b.sigma &&
            a.epsilon == b.epsilon, "update changed a 1-4 parameter or scale");
  for (auto [index, row] : llvm::enumerate(changed.dihedrals)) {
    const auto &old = original->dihedrals[index];
    require(row.i == old.i && row.j == old.j && row.k == old.k && row.l == old.l &&
            row.n == old.n && row.phase == old.phase && row.improper == old.improper,
            "update changed structural torsion metadata");
    if (row.improper)
      require(row.force == old.force, "update changed an improper coefficient");
  }
  auto after = take(driver::buildProgram(prepared.control, updated));
  require(before.module == after.module, "coefficient values changed generated IR");
  for (auto [a, b] : llvm::zip_equal(before.tupleSets, after.tupleSets)) {
    require(a.name == b.name && a.members == b.members, "tuple order changed");
    for (auto [field, other] : llvm::zip_equal(a.fields, b.fields))
      if (a.name != "dihedrals" || field.name != "k")
        require(field.values == other.values, "update changed another tuple field");
  }
  auto poisoned = prepared.system;
  model::poisonTunables(prepared.tunables, prepared.control, poisoned);
  auto arrays = take(driver::buildArrays(prepared.control, poisoned));
  require(driver::keepArraysOf(*arrays, before), "poisoned array structure changed");
  auto [kept, made] = driver::countArrays(*arrays);
  require(kept > 0 && made == 1, "coefficient update must make exactly one array");
  auto refilled = take(driver::buildProgram(prepared.control, updated, arrays.get()));
  require(driver::describeDifference(after, refilled).empty(),
          "retained-array refill differs from complete construction");
  llvm::outs() << "dihedral arrays: 1-4 parameters/scales, exclusions, row identity and "
                  "fixed fields unchanged; exactly one coefficient array rebuilt\n";
}
