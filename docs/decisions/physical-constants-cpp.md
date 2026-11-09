# D[physical-constants-cpp]

Recorded 2026-11-09.

Part 1 of #805. `include/mdir/Support/PhysicalConstants.h` is the single
definition site for physical inputs consumed by C++ and the plain-C
runtime. Its only physical inputs are the exact SI defining constants
$e$, $k$ and $N_A$ of [[BIPM2019]](../references.md#bipm2019), and the
nominal vacuum permittivity $8.8541878188\times10^{-12}$ F/m from CODATA
2022 [[Mohr2025]](../references.md#mohr2025). C++ uses named constants;
the C runtime uses the same header's macros. Pi is rounded to f64;
unit conversions use exact decimal definitions, including 4.184 J/cal.

Coulomb derives as $N_Ae^2/(4\pi\varepsilon_0)$, then converts from
J m/mol to native kJ nm/mol and control kcal Å/mol units. The compiler
rounds the stated f64 expression to 138.93545755023302 and
332.06371307417066, respectively. The molar gas coefficient derives from
$kN_A$; pressure and molar mass-density conversions derive from $N_A$.
No rounded native compatibility coefficient is retained.

`coulombInternal`, `units::coulomb` and `units::boltzmann` become aliases.
Generalized Born screening consumes the same derived Coulomb and gas
coefficients, with the salt/length conversion in the header. Native pressure and density coefficients,
and the readers' conversion definitions use the public header as well.
No control key, file format or overwrite behavior changes.

## Output compatibility

Native and control-expression outputs change bitwise wherever the new
coefficients enter; the maintainer accepted this before 1.0.0. Native
Coulomb changes by approximately −6.75e-10 relative; control-expression
Coulomb changes by approximately −6.78e-10. Thermal factors, velocity
initialization, pressure, density and salted Born screening can also
change. Generated programs, numerical reference outputs and checkpoints
therefore need not be bitwise those of the preceding constants.
The symmetry probe in the control reader uses the same coefficient; it
does not enter a calculated physical result.

## Maintainer's precision ruling

The first draft proposed retaining rounded native coefficients. The
maintainer superseded that choice on #807: every coefficient instead
derives from the exact SI inputs and CODATA 2022 permittivity, with one
value in a run. Native bit changes are accepted. The molar thermal
coefficient is 23 ulps above its preceding rounded value. Reference and
corpus regeneration records these intentional changes without changing
the comparison tolerances. Comparisons with other engines still name
their own constants; the ratio to sander's constant is updated.

## Validation

The C/C++ fixture requires identical coefficient bits in both languages.
Its independent 80-digit Decimal formulas read the defining inputs from
the public header instead of duplicating physical numbers. For unit
roundoff $u=2^{-53}$, a chain with at most $n$ input/arithmetic rounding
factors is bounded by $\gamma_n=nu/(1-nu)$ relative. The declared budgets
are 12/16 factors for native/control Coulomb, 4/6 for native/control gas,
6/8 for bar/atm pressure, 3 for density, and 24 for the longest Debye
derivation through Coulomb and gas coefficients. A defining input has a
one-factor bound. This tests the compiler-rounded expression against the
high-precision physical formula; it does not fit a new decimal literal.

Verify both added references, regenerate affected numerical references
and the corpus, classify changes against the base, and run the full
suite at the final clean head. No final results are claimed yet. Python
consolidation and the generated-module/duplicate-check work belong to
part 2 of #805. #795 rebases onto the merged header before its oracle
includes it; its decision names the header/conventions rather than a
literal and its stopped case is then rerun once.
