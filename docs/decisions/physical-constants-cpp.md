# D[physical-constants-cpp]

Recorded 2026-11-09.

Part 1 of #805. `include/mdir/Support/PhysicalConstants.h` is the single
definition site for physical inputs consumed by C++ and the plain-C
runtime. The source is CODATA 2018
[[Tiesinga2021]](../references.md#tiesinga2021). C++ uses named constants;
the C runtime uses the same header's macros.

The primary Coulomb coefficient is 138.935457644 kJ nm mol⁻¹ e⁻², the
documented native-topology value. The control-expression coefficient is
derived by dividing by the exact conversion factors 4.184 kJ/kcal and
0.1 nm/Å. This gives 332.06371329827908 kcal Å mol⁻¹ e⁻².

`coulombInternal`, `units::coulomb` and `units::boltzmann` become aliases.
The generalized Born SI inputs, native pressure and density coefficients,
and the readers' conversion definitions use the public header as well.
No control key, file format or overwrite behavior changes.

## Output compatibility

Native Coulomb, thermal, pressure, density and SI coefficients retain
their existing f64 bit patterns in this draft. No native-topology program
or checkpoint is intended to change. Control-expression programs that
name `coulomb` change by approximately 2.75e-12 relative in that input;
this is an intentional correction to one documented coefficient.
The symmetry probe in the control reader uses the same coefficient; it
does not enter a calculated physical result.

## Open precision choice

The existing native molar thermal coefficient is
0.0083144626181532004 kJ mol⁻¹ K⁻¹ as an f64 value. Deriving it from the
exact SI inputs gives 0.0083144626181532403, **23 ulps higher**. Likewise,
recomputing the native pressure conversion from the exact Avogadro
constant would change its bits. These cannot both be recomputed at full
precision and leave every native output bitwise unchanged.

This draft preserves the published native molar and pressure coefficients
as rounded primary model inputs, separately from the exact SI Boltzmann
and Avogadro inputs used by the Debye formula. Other units derive from
their respective primary values. Approval of that explicit rounding
contract is pending before validation. The alternative is to derive
those native coefficients at full SI precision and explicitly authorize
the resulting native output changes.

## Validation

Before ready, check the header from C and C++, native coefficient bits,
control-expression coefficients and affected references. Compare native
programs against the PR base bitwise, isolate expected control-expression
differences, and run the full suite at the final clean head. No results
are claimed yet. Python copies and the generated-module/duplicate-check
work belong to part 2 of #805. #795 rebases onto the merged header before
its oracle uses it and its stopped case is rerun.
