### Changed

- The TOWERS force loop converts a central pair force with one selected and
  scaled radial factor and screens the three scaled components with one sum
  of squares; the exact endpoint comparisons run only when that test fails.
  Force words and failure votes are unchanged (D[towers-pair-tail], #954).
