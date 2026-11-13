// The generator of the segments of rigid kinds (D405) on a real
// system: the kinds of D403 from its topology, and the pair term's primary
// kernels from the program that `mdir emit --stage=module` writes for it.
//
//   mdir-kind-pairs-test generate RUN.toml MODULE.mlir KINDS EXTENTS [--entry]
//   mdir-kind-pairs-test refuse RUN.toml KINDS EXTENTS
//
// KINDS lists the numbers of the kinds that are taken (as `mdir check`
// numbers them, from 1), EXTENTS their covering radii in nm, both
// comma-separated. `generate` replaces the primary pair kernel of the
// potentials @energy, @term_lj, and @term_coulomb by the segments' term and
// prints a module of them; with --entry, of @energy and a function that
// evaluates its energy, forces, and virial. `refuse` prints the reason that
// the plan is refused, or fails.
#include "../../lib/Driver/KindPairsIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/ProgramDialects.h"
#include "mdir/Driver/Control.h"
#include "mdir/Driver/RigidGroups.h"
#include "mdir/Driver/System.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

using namespace mlir;
using namespace mdir;
using namespace mdir::driver;

[[noreturn]] static void fail(const llvm::Twine &message) {
  llvm::errs() << "mdir-kind-pairs-test: " << message << "\n";
  std::exit(1);
}

static SmallVector<StringRef> split(StringRef list) {
  SmallVector<StringRef> items;
  if (!list.empty()) list.split(items, ',');
  return items;
}

/// The plan of the kinds that the arguments name, from the run's topology.
static llvm::Expected<KindPairsPlan> plan(StringRef control, StringRef kinds,
                                          StringRef extents, RigidGroups &groups,
                                          System &system) {
  auto read = readControl(control);
  if (!read) fail(llvm::toString(read.takeError()));
  auto made = readSystem(*read);
  if (!made) fail(llvm::toString(made.takeError()));
  system = std::move(*made);
  if (!system.topology) fail("the run has no topology");
  groups = findRigidGroups(*system.topology);
  SmallVector<const RigidKind *> taken;
  SmallVector<double> radii;
  for (StringRef item : split(kinds)) {
    unsigned number = 0;
    if (item.getAsInteger(10, number) || number == 0 || number > groups.kinds.size())
      fail("no kind " + item);
    taken.push_back(&groups.kinds[number - 1]);
  }
  for (StringRef item : split(extents)) {
    double radius = 0.0;
    if (item.getAsDouble(radius)) fail("not a radius: " + item);
    radii.push_back(radius);
  }
  return planKindPairs(*system.topology, taken, radii);
}

/// The primary pair kernel of a potential: the md.sum_relation over the
/// pairs of md.neighborhood.
static md::SumRelationOp findPrimary(md::PotentialOp potential) {
  md::SumRelationOp found;
  potential.walk([&](md::SumRelationOp sum) {
    if (sum.getRelation().getDefiningOp<md::NeighborhoodOp>() && !found) found = sum;
  });
  return found;
}

/// A copy of another potential's primary kernel at the start of `into`,
/// its captured arguments those of `into` (the potentials of a program
/// share their signature).
static md::SumRelationOp copyPrimary(md::SumRelationOp source, md::PotentialOp into) {
  auto from = source->getParentOfType<md::PotentialOp>();
  Block &target = into.getBody().front();
  if (from.getBody().front().getNumArguments() != target.getNumArguments())
    fail("the potentials do not share their signature");
  IRMapping mapping;
  for (auto [old, value] : llvm::zip(from.getBody().front().getArguments(),
                                     target.getArguments()))
    mapping.map(old, value);
  // The kernel's relation is not read by the clone of its body.
  OpBuilder builder = OpBuilder::atBlockBegin(&target);
  auto neighborhood = source.getRelation().getDefiningOp();
  mapping.map(source.getRelation(), builder.clone(*neighborhood, mapping)->getResult(0));
  return cast<md::SumRelationOp>(builder.clone(*source, mapping));
}

int main(int argc, char **argv) {
  if (argc < 5) fail("usage: generate|refuse RUN.toml [MODULE.mlir] KINDS EXTENTS");
  StringRef mode = argv[1];
  RigidGroups groups;
  System system;
  if (mode == "refuse") {
    auto made = plan(argv[2], argv[3], argv[4], groups, system);
    if (made) fail("the plan was not refused");
    llvm::outs() << "refused: " << llvm::toString(made.takeError()) << "\n";
    return 0;
  }
  if (mode != "generate" || argc < 6) fail("unknown mode " + mode);
  bool entry = argc > 6 && StringRef(argv[6]) == "--entry";
  auto made = plan(argv[2], argv[4], argv[5], groups, system);
  if (!made) fail(llvm::toString(made.takeError()));
  KindPairsPlan &kinds = *made;

  DialectRegistry registry;
  insertProgramDialects(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = parseSourceFile<ModuleOp>(argv[3], &context);
  if (!module) fail("cannot read the module");
  auto lookup = [&](StringRef name) {
    return module->lookupSymbol<md::PotentialOp>(name);
  };
  md::PotentialOp energy = lookup("energy"), lj = lookup("term_lj"),
                  coulomb = lookup("term_coulomb");
  if (!energy) fail("the module has no @energy");
  // Which pair kernels each potential takes: Lennard-Jones (1), Coulomb (2)
  // or both (3), each from the potential that computes that part alone.
  struct Target {
    md::PotentialOp potential;
    unsigned enabled;
  };
  SmallVector<Target> targets = {{energy, 3}};
  if (!entry) {
    if (lj) targets.push_back({lj, 1});
    if (coulomb) targets.push_back({coulomb, 2});
  }
  std::array<md::SumRelationOp, 4> sources = {
      {{}, lj ? findPrimary(lj) : md::SumRelationOp(),
       coulomb ? findPrimary(coulomb) : md::SumRelationOp(), findPrimary(energy)}};

  OpBuilder declarations = OpBuilder::atBlockBegin(module->getBody());
  for (Operation &op : *module->getBody())
    if (isa<md::TupleSetOp>(op)) declarations.setInsertionPointAfter(&op);
  builder::declareKindPairsTupleSets(declarations, kinds, "atoms");

  for (Target &target : targets) {
    md::PotentialOp potential = target.potential;
    md::SumRelationOp primary = findPrimary(potential);
    if (!primary) fail("a potential has no primary pair kernel");
    // These temporary kernels give the existing scalar expressions only.
    std::array<md::SumRelationOp, 4> kernels;
    for (unsigned mask = 1; mask != 4; ++mask) {
      if ((mask & target.enabled) != mask) continue;
      if (mask == target.enabled) kernels[mask] = primary;
      else if (sources[mask]) kernels[mask] = copyPrimary(sources[mask], potential);
      else fail("no pair kernel for a part of the pair term");
    }
    Block &body = potential.getBody().front();
    builder::KindPairsInputs input;
    for (unsigned f = 0; f != kinds.kinds.size(); ++f)
      input.kinds.push_back(body.addArgument(
          md::RelationType::get(&context,
                                cast<md::FieldType>(primary.getPositions().getType())
                                    .getParticleSet(),
                                kinds.kinds[f].getSize(), md::Orientation::Ordered,
                                FlatSymbolRefAttr::get(&context,
                                                       builder::getKindTupleSetName(f))),
          potential.getLoc()));
    potential.setFunctionTypeAttr(TypeAttr::get(FunctionType::get(
        &context, body.getArgumentTypes(), potential.getFunctionType().getResults())));
    auto neighborhood = primary.getRelation().getDefiningOp<md::NeighborhoodOp>();
    input.excluded = neighborhood.getExcluded();
    input.positions = primary.getPositions();
    input.cell = primary.getCell();
    auto gathered = primary.getGathered();
    input.types = gathered[0];
    input.charges = gathered[1];
    Block &kernel = primary.getKernel().front();
    if (kernel.getNumArguments() == 8) input.alchemical = gathered[2];
    input.cutoff = neighborhood.getCutoff().convertToDouble();
    input.lennardJones = target.enabled & 1;
    input.coulomb = target.enabled & 2;
    OpBuilder at(primary);
    Value term = builder::buildKindPairsTerm(
        at, kinds, input,
        [&](OpBuilder &ir, const builder::KindSitePair &pair) -> Value {
          unsigned mask = unsigned(pair.lennardJones) | (unsigned(pair.coulomb) << 1);
          return builder::cloneKindPairEnergy(ir, kernels[mask].getKernel().front(), pair);
        });
    primary.getResult().replaceAllUsesWith(term);
    for (auto kernel : kernels)
      if (kernel) {
        Operation *relation = kernel.getRelation().getDefiningOp();
        kernel.erase();
        if (relation->use_empty()) relation->erase();
      }
  }
  // Keep the declarations and the potentials that were generated.
  for (Operation &op : llvm::make_early_inc_range(*module->getBody())) {
    if (isa<md::ParticleSetOp, md::TupleSetOp>(op)) continue;
    if (llvm::any_of(targets, [&](Target &t) { return t.potential == &op; })) continue;
    op.erase();
  }
  if (entry) {
    // energy, forces and virial of @energy, at the IR level.
    OpBuilder at = OpBuilder::atBlockEnd(module->getBody());
    auto types = energy.getFunctionType().getInputs();
    Type positions = types[0];
    Type virial = VectorType::get({9}, at.getF64Type());
    auto function = md::FunctionOp::create(
        at, energy.getLoc(), at.getStringAttr("entry"),
        TypeAttr::get(FunctionType::get(&context, types,
                                        {at.getF64Type(), positions, virial})),
        StringAttr(), ArrayAttr(), ArrayAttr());
    Block *block = function.addEntryBlock();
    at.setInsertionPointToEnd(block);
    auto evaluate = md::EvaluateOp::create(
        at, energy.getLoc(), TypeRange{at.getF64Type(), positions, virial}, "energy",
        block->getArguments(),
        ArrayRef<int32_t>{int32_t(md::Request::Energy), int32_t(md::Request::Forces),
                          int32_t(md::Request::Virial)},
        ArrayRef<int64_t>{0, 0, 0});
    md::ReturnOp::create(at, energy.getLoc(), evaluate.getResults());
  }
  if (failed(verify(*module))) fail("the generated module does not verify");
  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
