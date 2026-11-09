// The plain-C runtime and the C++ compiler must consume one set of bits.
#include "mdir/Support/PhysicalConstants.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

extern "C" void mdirTestPhysicalConstantsC(double *values);

int main() {
  using namespace mdir::physical;
  std::array<double,5> c{}, cpp{coulomb,molarGasConstant,pressureBar,
      densityGramsPerCubicCentimeter,debyeSquaredMolarCoefficient};
  mdirTestPhysicalConstantsC(c.data());
  if (std::memcmp(c.data(),cpp.data(),sizeof(c)) != 0) {
    std::cerr << "C and C++ physical coefficients differ at the bit level\n";
    return 1;
  }
  std::cout << std::setprecision(17);
  auto emit=[](const char *name,double value) {
    uint64_t bits;
    std::memcpy(&bits,&value,sizeof(value));
    std::cout << name << ' ' << value << ' ' << bits << '\n';
  };
  emit("coulomb",coulomb);emit("coulomb_control",coulombControl);
  emit("gas",molarGasConstant);emit("gas_control",molarGasConstantControl);
  emit("pressure_bar",pressureBar);emit("pressure_atm",pressureAtmosphere);
  emit("density",densityGramsPerCubicCentimeter);
  emit("debye",debyeSquaredMolarCoefficient);
  emit("boltzmann",boltzmann);emit("avogadro",avogadro);
  emit("charge",elementaryCharge);emit("permittivity",vacuumPermittivity);
  emit("pi",MDIR_PHYSICAL_PI);
  emit("kcal",kilojoulesPerKilocalorie);
  emit("angstrom",nanometersPerAngstrom);emit("atmosphere",barPerAtmosphere);
}
