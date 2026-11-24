// Layout of the radial polynomial transform. Below the transition use a Hermite
// table in angular frequency. Beyond it the endpoint expansion is exact
// for the even Legendre polynomial of degree at most 78.
#pragma once

namespace mdir::prolate {
inline constexpr int SpectrumBasis = 40;
inline constexpr int SpectrumTransition = 512;
inline constexpr int SpectrumSamplesPerUnit = 16;
inline constexpr int SpectrumKnots = SpectrumTransition * SpectrumSamplesPerUnit + 1;
inline constexpr int SpectrumTerms = SpectrumKnots + SpectrumBasis;
// Three metadata words and the transform and window coefficients for values and slopes.
inline constexpr int SpectrumDeviceTableWords = 3 + 2 * SpectrumTerms + 2 * 8 * 64;
}
