#include "InitialVelocityDraws.h"
#include <array>
#include <cstring>

int main() {
  constexpr size_t count = 73;
  using Row = std::array<double, 3>;
  std::array<Row, count> reference;
  mdir::driver::InitialVelocityDraws serial(314159), reordered(314159);
  for (Row &row : reference)
    for (double &v : row)
      v = serial.nextNormal();
  for (size_t i = 0; i < count; ++i) {
    size_t id = (17 * i) % count;
    reordered.particle(id);
    Row row;
    for (double &v : row)
      v = reordered.nextNormal();
    if (std::memcmp(row.data(), reference[id].data(), sizeof(row))) return 1;
  }
}
