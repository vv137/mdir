// Physical inputs of MDIR, from CODATA 2018:
// E. Tiesinga et al., Rev. Mod. Phys. 93, 025010 (2021),
// doi:10.1103/RevModPhys.93.025010; docs/conventions.md, Section 1.
// Keep the published native coefficients bitwise unchanged. SI inputs
// used by the Debye formula are distinct from the rounded molar inputs.
// C macros also serve the plain-C runtime; C++ clients use the aliases.
#ifndef MDIR_SUPPORT_PHYSICALCONSTANTS_H
#define MDIR_SUPPORT_PHYSICALCONSTANTS_H

#define MDIR_PHYSICAL_COULOMB_NATIVE 138.935457644
#define MDIR_PHYSICAL_MOLAR_GAS_CONSTANT 0.0083144626181532
#define MDIR_PHYSICAL_BOLTZMANN_SI 1.380649e-23
#define MDIR_PHYSICAL_AVOGADRO 6.02214076e23
#define MDIR_PHYSICAL_ELEMENTARY_CHARGE 1.602176634e-19
#define MDIR_PHYSICAL_VACUUM_PERMITTIVITY 8.8541878128e-12
#define MDIR_PHYSICAL_PRESSURE_BAR 16.6053906717
#define MDIR_PHYSICAL_DENSITY_G_PER_CM3 1.66053906660e-3
#define MDIR_PHYSICAL_KJ_PER_KCAL 4.184
#define MDIR_PHYSICAL_NM_PER_ANGSTROM 0.1
#define MDIR_PHYSICAL_BAR_PER_ATM 1.01325

#ifdef __cplusplus
namespace mdir::physical {
// kJ nm mol^-1 e^-2. Primary Coulomb input; native programs retain its bits.
inline constexpr double coulomb = MDIR_PHYSICAL_COULOMB_NATIVE;
inline constexpr double kilojoulesPerKilocalorie = MDIR_PHYSICAL_KJ_PER_KCAL;
inline constexpr double nanometersPerAngstrom = MDIR_PHYSICAL_NM_PER_ANGSTROM;
inline constexpr double barPerAtmosphere = MDIR_PHYSICAL_BAR_PER_ATM;
// kcal Angstrom mol^-1 e^-2, derived by the exact unit conversions.
inline constexpr double coulombControl =
    coulomb / (kilojoulesPerKilocalorie * nanometersPerAngstrom);
// kJ mol^-1 K^-1: existing rounded native molar thermal coefficient.
inline constexpr double molarGasConstant = MDIR_PHYSICAL_MOLAR_GAS_CONSTANT;
inline constexpr double molarGasConstantControl =
    molarGasConstant / kilojoulesPerKilocalorie;
// SI inputs: J K^-1, mol^-1, C, and F m^-1, respectively.
inline constexpr double boltzmann = MDIR_PHYSICAL_BOLTZMANN_SI;
inline constexpr double avogadro = MDIR_PHYSICAL_AVOGADRO;
inline constexpr double elementaryCharge = MDIR_PHYSICAL_ELEMENTARY_CHARGE;
inline constexpr double vacuumPermittivity = MDIR_PHYSICAL_VACUUM_PERMITTIVITY;
// Existing native conversion coefficients, retained without recomputation.
inline constexpr double pressureBar = MDIR_PHYSICAL_PRESSURE_BAR;
inline constexpr double pressureAtmosphere = pressureBar / barPerAtmosphere;
inline constexpr double densityGramsPerCubicCentimeter = MDIR_PHYSICAL_DENSITY_G_PER_CM3;
} // namespace mdir::physical
#endif
#endif
