// Layout of the radial polynomial transform. Below the transition use a Hermite
// table in angular frequency. Beyond it the endpoint expansion is exact
// for the even Legendre polynomial of degree at most 78.
#pragma once

#ifdef __cplusplus
namespace mdir::prolate {
#define MDIR_SPECTRUM_VALUE(name, value) inline constexpr int name = value
#else
#define MDIR_SPECTRUM_VALUE(name, value) enum { name = value }
#endif
MDIR_SPECTRUM_VALUE(SpectrumBasis, 40);
MDIR_SPECTRUM_VALUE(SpectrumTransition, 512);
MDIR_SPECTRUM_VALUE(SpectrumSamplesPerUnit, 16);
MDIR_SPECTRUM_VALUE(SpectrumKnots, SpectrumTransition * SpectrumSamplesPerUnit + 1);
MDIR_SPECTRUM_VALUE(SpectrumTerms, SpectrumKnots + SpectrumBasis);
// Three metadata words and the transform and window coefficients for values and slopes.
MDIR_SPECTRUM_VALUE(SpectrumDeviceTableWords, 3 + 2 * SpectrumTerms + 2 * 8 * 64);
#undef MDIR_SPECTRUM_VALUE
#ifdef __cplusplus
}
#endif
