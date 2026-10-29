### Changed

- Construct term fragments and custom-expression scalar kernels with typed `OpBuilder` operations at their shared potential insertion sites, preserving generated programs and loop fusion (D[builder-term-ir], #568, #708).
- Make `mdir emit` report an internal construction error instead of emitting raw text when generated assembly does not parse, amending D352 because its silent fallback hid #706 and #714 (D[builder-term-ir], #709).
- Replace the narrow hosted-model emission stream with a synchronous typed-fragment writer while retaining read-only inputs and owned recipes (D[builder-term-ir], #568).
