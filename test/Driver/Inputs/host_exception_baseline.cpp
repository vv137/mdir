// Establish the toolchain's exception interceptor before testing a JIT.
#include <stdexcept>
#include <iostream>
int main() {
  for (int i = 0; i != 32; ++i) {
    try { throw std::runtime_error("missing input"); }
    catch (const std::runtime_error &) {}
  }
  std::cout << "host exception baseline: 32 throws, 0 failures\n";
}
