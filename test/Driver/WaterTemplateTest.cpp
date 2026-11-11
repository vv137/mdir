// Boundary cases of the topology proof, independent of a kernel or GPU.
#include "mdir/Driver/WaterTemplate.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

using namespace mdir::driver;

static void require(bool value, const char *message) {
  if (!value) { llvm::errs() << message << '\n'; std::exit(1); }
}
static System input(unsigned four = 2, unsigned three = 1) {
  System s;
  s.topology = std::make_shared<Topology>();
  auto &t = *s.topology;
  t.typeNames = {"O", "H", "M", "ion"};
  t.sigma.assign(16, 0.0); t.epsilon.assign(16, 0.0);
  t.sigma[0] = 0.32; t.epsilon[0] = 0.7;
  auto water = [&](unsigned count) {
    unsigned o = t.getNumParticles();
    t.settles.push_back({o, 0.1, 0.16});
    t.residueStarts.push_back(o); t.residueNames.push_back("WAT");
    for (unsigned k = 0; k != count; ++k) {
      t.atomicNumbers.push_back(k == 0 ? 8 : k == 3 ? 0 : 1);
      t.atomNames.push_back(k == 0 ? "O" : k == 3 ? "M" : "H");
      t.masses.push_back(k == 0 ? 16.0 : k == 3 ? 0.0 : 1.0);
      t.charges.push_back(k == 0 ? (count == 3 ? -1.0 : 0.0) : k == 3 ? -1.0 : 0.5);
      t.types.push_back(k == 0 ? 0 : k == 3 ? 2 : 1);
      t.residueOf.push_back(t.residueNames.size() - 1);
      for (unsigned j = 0; j != k; ++j) t.exclusions.emplace_back(o + j, o + k);
    }
    if (count == 4) t.virtualSites.push_back({Topology::VirtualSite::AmberWater, o + 3, o, o + 1, o + 2, 0.02, 0.0});
  };
  for (unsigned i = 0; i != four; ++i) water(4);
  for (unsigned i = 0; i != three; ++i) water(3);
  t.residueStarts.push_back(t.getNumParticles()); t.residueNames.push_back("ION");
  t.residueOf.push_back(t.residueNames.size() - 1);
  t.atomicNumbers.push_back(19); t.atomNames.push_back("K");
  t.masses.push_back(39.0); t.charges.push_back(1.0); t.types.push_back(3);
  s.types = t.types; s.masses = t.masses;
  return s;
}
/// A methyl-like solute fragment: a carbon and three hydrogens, with or
/// without the SHAKE group of its X-H bonds.
static void addMethyl(System &s, bool constrained) {
  auto &t = *s.topology;
  unsigned c = t.getNumParticles();
  t.residueStarts.push_back(c); t.residueNames.push_back("MET");
  for (unsigned k = 0; k != 4; ++k) {
    t.atomicNumbers.push_back(k ? 1 : 6); t.atomNames.push_back(k ? "H" : "C");
    t.masses.push_back(k ? 1.008 : 12.0); t.charges.push_back(k ? 0.1 : -0.3);
    t.types.push_back(3); t.residueOf.push_back(t.residueNames.size() - 1);
    if (k) t.exclusions.emplace_back(c, c + k);
  }
  if (constrained) t.shakes.push_back({c, {c + 1, c + 2, c + 3}, {0.109, 0.108, 0.109}});
  s.types = t.types; s.masses = t.masses;
}
static WaterTemplatePlan match(const System &s, const Control &c = {}) {
  auto plan = matchWaterTemplate(c, s);
  if (!plan) { llvm::errs() << llvm::toString(plan.takeError()) << '\n'; std::exit(1); }
  return std::move(*plan);
}
int runWaterTemplateIRTest();
int runWaterTemplateUnitIRTest();
int runWaterTemplateLayoutTest();
int runWaterTemplateValidityTest();
int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "--validity-ir") == 0)
    return runWaterTemplateValidityTest();
  if (argc == 2 && std::strcmp(argv[1], "--layout") == 0)
    return runWaterTemplateLayoutTest();
  if (argc == 2 && std::strcmp(argv[1], "--window-ir") == 0)
    return runWaterTemplateIRTest();
  if (argc == 2 && std::strcmp(argv[1], "--unit-ir") == 0)
    return runWaterTemplateUnitIRTest();
  if (argc == 3 && std::strcmp(argv[1], "--select") == 0) {
    // The route that a control file's system takes, and why.
    auto c = readControl(argv[2]);
    if (!c) { llvm::errs() << llvm::toString(c.takeError()) << '\n'; return 1; }
    auto s = readSystem(*c);
    if (!s) { llvm::errs() << llvm::toString(s.takeError()) << '\n'; return 1; }
    auto selected = selectWaterTemplate(*c, *s, 0);
    if (!selected) { llvm::errs() << llvm::toString(selected.takeError()) << '\n'; return 1; }
    llvm::outs() << "water.template route: " << (selected->enabled ? "template" : "general")
                 << " (" << selected->reason << ")\n";
    return 0;
  }
  if (argc == 2) {
    auto c = readControl(argv[1]);
    if (!c) { llvm::errs() << llvm::toString(c.takeError()) << '\n'; return 1; }
    const char *name = c->waterTemplate == WaterTemplateChoice::Auto ? "AUTO" :
                       c->waterTemplate == WaterTemplateChoice::On ? "ON" : "OFF";
    llvm::outs() << "water.template = " << name << '\n';
    return 0;
  }
  unsigned cases = 0;
  {
    auto s = input(1, 0);
    s.topology->virtualSites[0].a = 0.09;
    auto p = match(s);
    require(p.instances.size() == 1 && p.model.sites == 4 &&
            std::abs(p.model.alpha - 0.75) < 1e-15 &&
            std::abs(p.model.extent - 0.1001) < 1e-15,
            "alpha 0.75 topology and formed-site covering bound");
    ++cases;
  }
  {
    auto p = match(input());
    require(p.candidates == 3 && p.instances.size() == 2 && p.model.sites == 4 && p.solute.size() == 4,
            "predominant template or solute partition");
    require(p.mismatches.at("other water template") == 1, "minority report");
    require(p.model.alpha > 0.1666666666666 && p.model.alpha < 0.1666666666668 &&
            std::abs(p.model.extent - 0.1001) < 1e-15, "geometry constants or margin units");
    ++cases;
  }
  {
    auto p = match(input(1, 1));
    require(p.model.sites == 4 && p.instances[0][0] == 0, "stable predominant tie"); ++cases;
  }
  {
    auto p = match(input(0, 1));
    require(p.model.sites == 3 && p.instances.size() == 1 && p.model.alpha == 0.0 && p.instances[0][3] == 0,
            "three-site template"); ++cases;
  }
  {
    auto s = input(); auto &t = *s.topology;
    t.settles.erase(t.settles.begin());
    t.bonds.push_back({0, 1, 1.0, 0.1, true}); t.bonds.push_back({0, 2, 1.0, 0.1, true});
    auto p = match(s);
    require(p.instances.size() == 1 && p.mismatches.at("flexible or partially constrained water") == 1,
            "flexible water proof"); ++cases;
  }
  {
    auto s = input(); s.topology->epsilon[5] = 0.1;
    auto p = match(s);
    require(p.instances.empty() && p.mismatches.at("LJ on the hydrogens") == 3, "hydrogen LJ"); ++cases;
  }
  {
    auto s = input(); s.topology->epsilon[8] = 0.1;
    auto p = match(s);
    require(p.model.sites == 3 && p.mismatches.at("LJ on the extra point") == 2, "extra-point LJ cross type"); ++cases;
  }
  {
    auto s = input(); Control c; c.pairs.push_back({});
    s.pairGroups = {{std::vector<bool>(s.types.size(), false), std::vector<bool>(s.types.size(), false)}};
    s.pairGroups[0][0][0] = true; s.pairGroups[0][1].back() = true;
    auto p = match(s, c);
    require(p.instances.size() == 1 && p.mismatches.at("custom pair or particle parameter") == 1,
            "custom group membership"); ++cases;
  }
  {
    Control c; c.pairs.push_back({}); auto p = match(input(), c);
    require(p.instances.empty() && p.mismatches.at("custom pair or particle parameter") == 3,
            "ungrouped custom pairs"); ++cases;
  }
  {
    Control c; ParticleParameter q; q.name = "zero"; q.particles = {0}; c.particleParameters.push_back(q);
    auto p = match(input(), c);
    require(p.instances.size() == 1 && p.mismatches.at("custom pair or particle parameter") == 1,
            "zero-valued particle parameter remains active metadata"); ++cases;
  }
  {
    auto s = input(); s.alchemical.assign(s.types.size(), false); s.alchemical[0] = true;
    auto p = match(s);
    require(p.instances.size() == 1 && p.mismatches.at("free-energy selection") == 1, "perturbed membership"); ++cases;
  }
  {
    auto s = input(); s.topology->exclusions.emplace_back(0, s.types.size() - 1);
    auto p = match(s);
    require(p.instances.size() == 1 && p.mismatches.at("intermolecular exclusion") == 1, "cross-molecule exclusion"); ++cases;
  }
  {
    auto s = input(); auto &e = s.topology->exclusions;
    e.erase(std::find(e.begin(), e.end(), std::make_pair(1u, 2u)));
    auto p = match(s);
    require(p.instances.size() == 1 && p.mismatches.at("incomplete intramolecular exclusions") == 1,
            "missing internal exclusion"); ++cases;
  }
  {
    auto s = input(); s.topology->virtualSites[0].kind = Topology::VirtualSite::Linear;
    auto p = match(s);
    require(p.instances.size() == 1 && p.mismatches.at("extra-point definition") == 1, "site definition"); ++cases;
  }
  {
    auto s = input(); auto &v = s.topology->virtualSites[0]; std::swap(v.j, v.k);
    require(match(s).instances.size() == 2, "symmetric bisector member order"); ++cases;
  }
  {
    auto s = input(); s.topology->charges[1] += 0.01;
    auto p = match(s);
    require(p.instances.size() == 1 && p.mismatches.at("other water template") == 2,
            "exact charge signature"); ++cases;
  }
  {
    Control c; c.ljpme = true; auto p = match(input(), c);
    require(p.instances.empty() && p.mismatches.at("LJ-PME") == 3, "LJ-PME"); ++cases;
  }
  {
    System s; s.types = {0}; auto p = match(s);
    require(p.instances.empty() && p.solute == std::vector<unsigned>{0}, "no-topology partition"); ++cases;
  }
  require(WaterTemplatePlan::mobilityMasks == std::array<unsigned, 3>{1, 1, 1}, "segment mobility incidence");
  {
    auto s = input(); addMethyl(s, true);
    auto p = match(s);
    unsigned c = s.types.size() - 4;
    // The three-site minority water is solute too: two more members.
    require(p.anchored.size() == 5 && p.unconstrainedHydrogens == 0 &&
            p.anchored[0] == std::array<unsigned, 2>{c + 1, c} &&
            p.anchored[2] == std::array<unsigned, 2>{c + 3, c} &&
            p.anchorLength == 0.109 &&
            std::abs(getAnchorRadius(p, false) - 0.109 * (1 + 1e-4)) < 1e-17 &&
            std::abs(getAnchorRadius(p, true) - 0.109 * (1 + 1e-8)) < 1e-17,
            "heavy-atom units from constrained X-H bonds"); ++cases;
  }
  {
    auto s = input(); addMethyl(s, false);
    auto p = match(s);
    require(p.anchored.size() == 2 && p.unconstrainedHydrogens == 3,
            "unconstrained solute hydrogens are counted, not anchored"); ++cases;
  }
  {
    // Unmatched rigid water stays solute: a unit of its oxygen.
    auto s = input(); s.topology->epsilon[5] = 0.1;
    auto p = match(s);
    require(p.instances.empty() && p.anchored.size() == 6 && p.anchorLength == 0.1 &&
            p.anchored[1] == std::array<unsigned, 2>{2, 0} && p.unconstrainedHydrogens == 0,
            "solute rigid water units"); ++cases;
  }
  llvm::outs() << "water template topology contracts: " << cases << " passed\n";
  Control c; c.target = Target::GPU; c.precision = Precision::Mixed;
  auto selection = [&](const System &s, size_t floor) {
    auto result = selectWaterTemplate(c, s, floor);
    if (!result) { llvm::errs() << llvm::toString(result.takeError()) << '\n'; std::exit(1); }
    return std::move(*result);
  };
  unsigned policies = 0;
  require(selection(input(), 0).enabled, "AUTO four-site eligibility"); ++policies;
  require(!selection(input(), 13).enabled, "AUTO measured floor"); ++policies;
  require(!selection(input(0, 2), 0).enabled, "AUTO three-site general path"); ++policies;
  c.waterTemplate = WaterTemplateChoice::On;
  require(selection(input(0, 2), 1000).enabled, "ON bypasses measured policy"); ++policies;
  auto refused = [&](const char *reason) {
    auto result = selectWaterTemplate(c, input(), 0);
    require(!result, "unsupported ON accepted");
    auto message = llvm::toString(result.takeError());
    require(message.find("water.template") != std::string::npos && message.find(reason) != std::string::npos,
            "named ON refusal");
    ++policies;
  };
  c.target = Target::CPU; refused("CPU"); c.target = Target::GPU;
  c.precision = Precision::Double; refused("MIXED"); c.precision = Precision::Mixed;
  c.deterministic = true; refused("deterministic"); c.deterministic = false;
  c.tunables = true; refused("tunable"); c.tunables = false;
  c.frames = true; refused("frame"); c.frames = false;
  c.minimize = true; refused("minimizes"); c.minimize = false;
  auto empty = input(0, 0); auto missing = selectWaterTemplate(c, empty, 0);
  require(!missing, "ON without matching water");
  require(llvm::toString(missing.takeError()).find("no rigid water") != std::string::npos, "no-match reason"); ++policies;
  {
    auto loose = input(); addMethyl(loose, false);
    auto result = selectWaterTemplate(c, loose, 0);
    require(!result, "ON accepted unconstrained solute hydrogens");
    require(llvm::toString(result.takeError()).find("3 solute hydrogens have no constrained bond") !=
            std::string::npos, "unconstrained-hydrogen ON refusal"); ++policies;
    auto held = input(); addMethyl(held, true);
    require(selection(held, 0).enabled, "ON with constrained solute units"); ++policies;
    c.waterTemplate = WaterTemplateChoice::Auto;
    auto general = selection(loose, 0);
    require(!general.enabled && general.reason.find("hydrogen_bonds") != std::string::npos,
            "AUTO takes the general path without constrained hydrogens"); ++policies;
    c.waterTemplate = WaterTemplateChoice::On;
  }
  c.waterTemplate = WaterTemplateChoice::Off; c.target = Target::CPU;
  require(!selection(input(), 0).enabled, "OFF ordinary path"); ++policies;
  c.waterTemplate = WaterTemplateChoice::Auto;
  require(!selection(input(), 0).enabled, "AUTO unsupported-mode ordinary path"); ++policies;
  llvm::outs() << "water template selection contracts: " << policies << " passed\n";
}
