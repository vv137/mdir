# D[distribute-mixed-displacement]

Recorded at its merge.

**Compensated binary32 midpoint displacements.**

The floating mixed midpoint row kernel splits each explicit-image
coordinate into a leading binary32 value and its binary32 residual.
Inside the pair loop, an error-free leading subtraction
[Higham2002] is corrected with the difference of the two residuals
before rounding the result in binary32. Staging once per row moves
binary64 coordinate work outside the pair loop. This uses
binary32 arithmetic for short-range displacements [Bowers2006].

The midpoint membership filter keeps its binary64 geometry. The
pair kernel applies its existing cutoff to the squared displacement.
Fixed-point and double force paths keep their original arithmetic.

[Higham2002]: ../references.md#higham2002
[Bowers2006]: ../references.md#bowers2006
