// Physical inputs: BIPM, The International System of Units, 9th ed. (2019),
// doi:10.59161/AUEZ1291 (exact e, k, N_A); Mohr et al., Rev. Mod. Phys.
// 97, 025002 (2025), doi:10.1103/RevModPhys.97.025002 (CODATA 2022 epsilon_0).
// Every derived coefficient is evaluated from these inputs at f64.
// C macros also serve the plain-C runtime; C++ clients use the aliases.
#ifndef MDIR_SUPPORT_PHYSICALCONSTANTS_H
#define MDIR_SUPPORT_PHYSICALCONSTANTS_H

#define MDIR_PHYSICAL_BOLTZMANN_SI 1.380649e-23
#define MDIR_PHYSICAL_AVOGADRO 6.02214076e23
#define MDIR_PHYSICAL_ELEMENTARY_CHARGE 1.602176634e-19
#define MDIR_PHYSICAL_VACUUM_PERMITTIVITY 8.8541878188e-12
// Mathematical pi rounded to f64, and exact unit definitions.
#define MDIR_PHYSICAL_PI 0x1.921fb54442d18p+1
#define MDIR_PHYSICAL_KJ_PER_KCAL 4.184
#define MDIR_PHYSICAL_NM_PER_ANGSTROM 0.1
#define MDIR_PHYSICAL_BAR_PER_ATM 1.01325
#define MDIR_PHYSICAL_COULOMB_NATIVE \
  (MDIR_PHYSICAL_AVOGADRO * MDIR_PHYSICAL_ELEMENTARY_CHARGE * \
   MDIR_PHYSICAL_ELEMENTARY_CHARGE / \
   (4.0 * MDIR_PHYSICAL_PI * MDIR_PHYSICAL_VACUUM_PERMITTIVITY) * 1e6)
#define MDIR_PHYSICAL_MOLAR_GAS_CONSTANT \
  (MDIR_PHYSICAL_BOLTZMANN_SI * MDIR_PHYSICAL_AVOGADRO / 1e3)
// kJ/mol per nm^3 -> J/particle per m^3 -> bar.
#define MDIR_PHYSICAL_PRESSURE_BAR \
  (1e3 / MDIR_PHYSICAL_AVOGADRO / 1e-27 / 1e5)
// Molar mass in g/mol per nm^3 -> g/cm^3.
#define MDIR_PHYSICAL_DENSITY_G_PER_CM3 (1e21 / MDIR_PHYSICAL_AVOGADRO)
#define MDIR_PHYSICAL_DEBYE_SQUARED \
  (8.0 * MDIR_PHYSICAL_PI * MDIR_PHYSICAL_COULOMB_NATIVE / \
   MDIR_PHYSICAL_MOLAR_GAS_CONSTANT * MDIR_PHYSICAL_AVOGADRO * 1e-24)

#ifdef __cplusplus
namespace mdir::physical {
// kJ nm mol^-1 e^-2, from N_A e^2/(4 pi epsilon_0).
inline constexpr double coulomb = MDIR_PHYSICAL_COULOMB_NATIVE;
inline constexpr double kilojoulesPerKilocalorie = MDIR_PHYSICAL_KJ_PER_KCAL;
inline constexpr double nanometersPerAngstrom = MDIR_PHYSICAL_NM_PER_ANGSTROM;
inline constexpr double barPerAtmosphere = MDIR_PHYSICAL_BAR_PER_ATM;
// kcal Angstrom mol^-1 e^-2, derived by the exact unit conversions.
inline constexpr double coulombControl =
    coulomb / (kilojoulesPerKilocalorie * nanometersPerAngstrom);
// kJ mol^-1 K^-1, from k N_A.
inline constexpr double molarGasConstant = MDIR_PHYSICAL_MOLAR_GAS_CONSTANT;
inline constexpr double molarGasConstantControl =
    molarGasConstant / kilojoulesPerKilocalorie;
// SI inputs: J K^-1, mol^-1, C, and F m^-1, respectively.
inline constexpr double boltzmann = MDIR_PHYSICAL_BOLTZMANN_SI;
inline constexpr double avogadro = MDIR_PHYSICAL_AVOGADRO;
inline constexpr double elementaryCharge = MDIR_PHYSICAL_ELEMENTARY_CHARGE;
inline constexpr double vacuumPermittivity = MDIR_PHYSICAL_VACUUM_PERMITTIVITY;
// Unit conversions derived from N_A.
inline constexpr double pressureBar = MDIR_PHYSICAL_PRESSURE_BAR;
inline constexpr double pressureAtmosphere = pressureBar / barPerAtmosphere;
inline constexpr double densityGramsPerCubicCentimeter = MDIR_PHYSICAL_DENSITY_G_PER_CM3;
// K nm^-2 per (mol/L): kappa^2 = coefficient * salt/(epsilon_r T).
inline constexpr double debyeSquaredMolarCoefficient = MDIR_PHYSICAL_DEBYE_SQUARED;
} // namespace mdir::physical
#endif
#endif
