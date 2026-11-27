#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mdir::driver {

/// A generator of random numbers whose sequence is fixed by its definition,
/// not by a library: splitmix64 for the state, xoshiro256** for the numbers.
/// See Steele et al., OOPSLA '14, 453 (2014), and Blackman and Vigna, ACM
/// Trans. Math. Softw. 47(4), 1 (2021); docs/references.md.
class InitialVelocityDraws {
  using State = std::array<uint64_t, 4>;
  using Transition = std::array<State, 256>;

public:
  explicit InitialVelocityDraws(uint64_t seed) {
    for (uint64_t &word : state) {
      seed += 0x9e3779b97f4a7c15ull;
      uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
      word = z ^ (z >> 31);
    }
    initial = state;
  }

  // Initial velocities use step zero. Six uniform words belong to each
  // particle. Seek to its counter even if the caller visits a different
  // order. Consecutive counters reuse the state left by the preceding row.
  void particle(std::size_t id) {
    uint64_t counter = uint64_t(id) * 6;
    if (counter == next)
      return;
    state = initial;
    next = counter;
    const auto &powers = transitionPowers();
    for (unsigned bit = 0; counter; counter >>= 1, ++bit)
      if (counter & 1)
        state = applyTransition(powers[bit], state);
  }

  /// A number in (0, 1].
  double nextUniform() {
    auto rotate = [](uint64_t x, int k) { return (x << k) | (x >> (64 - k)); };
    uint64_t result = rotate(state[1] * 5, 7) * 9;
    advance(state);
    ++next;
    return (static_cast<double>(result >> 11) + 1.0) * 0x1.0p-53;
  }

  /// A number from the normal distribution, by the method of Box and
  /// Muller, Ann. Math. Stat. 29, 610 (1958).
  double nextNormal() {
    double radius = std::sqrt(-2.0 * std::log(nextUniform()));
    double angle = 6.283185307179586 * nextUniform();
    return radius * std::cos(angle);
  }

private:
  static void advance(State &state) {
    uint64_t t = state[1] << 17;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = (state[3] << 45) | (state[3] >> 19);
  }

  // The xoshiro state transition is linear over GF(2). Its powers
  // locate any counter in the original trajectory without changing a draw.
  static State applyTransition(const Transition &matrix, const State &x) {
    State result{};
    for (unsigned i = 0; i < 256; ++i)
      if ((x[i / 64] >> (i % 64)) & 1)
        for (unsigned j = 0; j < 4; ++j)
          result[j] ^= matrix[i][j];
    return result;
  }
  static const std::array<Transition, 64> &transitionPowers() {
    static const auto powers = [] {
      std::array<Transition, 64> result{};
      for (unsigned i = 0; i < 256; ++i) {
        result[0][i][i / 64] = uint64_t(1) << (i % 64);
        advance(result[0][i]);
      }
      for (unsigned n = 1; n < result.size(); ++n)
        for (unsigned i = 0; i < 256; ++i)
          result[n][i] = applyTransition(result[n - 1], result[n - 1][i]);
      return result;
    }();
    return powers;
  }
  State initial, state;
  uint64_t next = 0;
};

} // namespace mdir::driver
