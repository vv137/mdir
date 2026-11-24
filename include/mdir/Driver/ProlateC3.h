// Layout of the radial C³ transform. Below the transition use a Hermite
// table in angular frequency. Beyond it the endpoint expansion is exact
// for the even Legendre polynomial of degree 78.
#pragma once

namespace mdir::prolate {
inline constexpr int C3Basis = 40;
inline constexpr int C3Transition = 512;
inline constexpr int C3SamplesPerUnit = 16;
inline constexpr int C3Knots = C3Transition * C3SamplesPerUnit + 1;
inline constexpr int C3Terms = C3Knots + C3Basis;
// Three metadata words and four cubic coefficients per point for values and slopes.
inline constexpr int C3DeviceTableWords = 3 + 2 * C3Terms + 2 * 4 * 4;
}
