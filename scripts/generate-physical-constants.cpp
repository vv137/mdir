// Emit the header's compiler-rounded values, not a second implementation
// of its formulas. Hexadecimal f64 literals preserve every bit in Python.
#include "mdir/Support/PhysicalConstants.h"
#include <iomanip>
#include <iostream>

int main() {
  using namespace mdir::physical;
  auto emit = [](const char *name, double value) {
    std::cout << name << " = float.fromhex(\"" << std::hexfloat << value
              << "\")\n";
  };
  emit("BOLTZMANN_SI", MDIR_PHYSICAL_BOLTZMANN_SI);
  emit("AVOGADRO", MDIR_PHYSICAL_AVOGADRO);
  emit("ELEMENTARY_CHARGE", MDIR_PHYSICAL_ELEMENTARY_CHARGE);
  emit("VACUUM_PERMITTIVITY", MDIR_PHYSICAL_VACUUM_PERMITTIVITY);
  emit("PI", MDIR_PHYSICAL_PI);
  emit("KJ_PER_KCAL", MDIR_PHYSICAL_KJ_PER_KCAL);
  emit("NM_PER_ANGSTROM", MDIR_PHYSICAL_NM_PER_ANGSTROM);
  emit("BAR_PER_ATM", MDIR_PHYSICAL_BAR_PER_ATM);
  emit("COULOMB_NATIVE", MDIR_PHYSICAL_COULOMB_NATIVE);
  emit("COULOMB_CONTROL", coulombControl);
  emit("MOLAR_GAS_CONSTANT", MDIR_PHYSICAL_MOLAR_GAS_CONSTANT);
  emit("MOLAR_GAS_CONSTANT_CONTROL", molarGasConstantControl);
  emit("PRESSURE_BAR", MDIR_PHYSICAL_PRESSURE_BAR);
  emit("PRESSURE_ATMOSPHERE", pressureAtmosphere);
  emit("DENSITY_G_PER_CM3", MDIR_PHYSICAL_DENSITY_G_PER_CM3);
  emit("DEBYE_SQUARED", MDIR_PHYSICAL_DEBYE_SQUARED);
}
