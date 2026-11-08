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
static WaterTemplatePlan match(const System &s, const Control &c = {}) {
  auto plan = matchWaterTemplate(c, s);
  if (!plan) { llvm::errs() << llvm::toString(plan.takeError()) << '\n'; std::exit(1); }
  return std::move(*plan);
}
int runWaterTemplateIRTest();
int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "--window-ir") == 0)
    return runWaterTemplateIRTest();
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
  require(WaterTemplatePlan::mobilityMasks == std::array<unsigned, 3>{1, 3, 2}, "segment mobility incidence");
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
  c.waterTemplate = WaterTemplateChoice::Off; c.target = Target::CPU;
  require(!selection(input(), 0).enabled, "OFF ordinary path"); ++policies;
  c.waterTemplate = WaterTemplateChoice::Auto;
  require(!selection(input(), 0).enabled, "AUTO unsupported-mode ordinary path"); ++policies;
  llvm::outs() << "water template selection contracts: " << policies << " passed\n";
}
