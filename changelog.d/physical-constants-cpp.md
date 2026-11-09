### Fixed

- Physical constants have one public C/C++
  definition, deriving Coulomb, thermal, pressure and density coefficients
  from the exact SI inputs and CODATA 2022 permittivity. Native and
  control-expression outputs change bitwise; the Coulomb input changes
  by about −6.75e-10 and −6.78e-10 relative, respectively. Debye screening
  uses the same derived constants (D[physical-constants-cpp], #805).
