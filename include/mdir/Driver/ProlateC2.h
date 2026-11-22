// Layout of the radial C² transform. Below the transition use a Hermite
// table in angular frequency. Beyond it the endpoint expansion is exact
// for the even Legendre polynomial of degree 78.
#pragma once

namespace mdir::prolate {
inline constexpr int C2PairCells = 1024;
inline constexpr int C2Basis = 40;
inline constexpr int C2Transition = 512;
inline constexpr int C2SamplesPerUnit = 16;
inline constexpr int C2Knots = C2Transition * C2SamplesPerUnit + 1;
inline constexpr int C2Terms = C2Knots + C2Basis;
// Three metadata words and four cubic coefficients per point for values and slopes.
inline constexpr int C2DeviceTableWords = 3 + 2 * C2Terms + 2 * 4 * 4;
}
