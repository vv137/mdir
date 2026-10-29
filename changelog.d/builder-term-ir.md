### Changed

- Construct term fragments and custom-expression scalar kernels with typed `OpBuilder` operations at their shared potential insertion sites, preserving generated programs and loop fusion (D363, #568, #708).
- Make `mdir emit` report an internal construction error instead of emitting raw text when generated assembly does not parse, amending D352 because its silent fallback hid #706 and #714 (D363, #709).
- Replace the narrow hosted-model emission stream with a synchronous typed-fragment writer while retaining read-only inputs and owned recipes (D363, #568).
