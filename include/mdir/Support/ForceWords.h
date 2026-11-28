// Signed force words in the moving pair and bonded work-unit kernels.
#ifndef MDIR_SUPPORT_FORCEWORDS_H
#define MDIR_SUPPORT_FORCEWORDS_H

namespace mdir {
/// Words per kJ mol^-1 nm^-1. Power-of-two scaling is exact before rounding.
inline constexpr int forceWordFractionBits = 18;
inline constexpr double forceWordScale = 1 << forceWordFractionBits;
/// Bonded energy words per kJ mol^-1.
inline constexpr double bondedEnergyScale = 0x1p32;
} // namespace mdir
#endif
