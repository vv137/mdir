# D[distribute-pencil-collectives]

Recorded at its merge.

**Scope FFT transposes to pencil peers.**

An adjacent transpose varies one coordinate of the pencil process grid
and keeps the other fixed. Each row or column of that grid now executes
its own variable-count all-to-all. The direct mixed inverse transpose
uses the whole mesh communicator because both coordinates change.
This adapts the axis-scoped FFT communication described by
[Jung2019Scaling] to a mesh that starts on pencils.

The runtime caches communicators for the selected process grid and rank
placement. Ranks outside the mesh skip FFT collectives and still
participate in charge and potential exchanges. Buffer layout, wire
precision, line order and arithmetic are unchanged.

[Jung2019Scaling]: ../references.md#jung2019scaling
