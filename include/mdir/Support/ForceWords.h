// Signed force words in the moving pair and bonded work-unit kernels.
#ifndef MDIR_SUPPORT_FORCEWORDS_H
#define MDIR_SUPPORT_FORCEWORDS_H

namespace mdir {
/// Words per kJ mol^-1 nm^-1. Power-of-two scaling is exact before rounding.
inline constexpr double forceWordScale = 0x1p18;
} // namespace mdir
#endif
