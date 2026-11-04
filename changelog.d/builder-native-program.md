### Changed

- Construct complete programs with owned native IR, parsing text only for
  user-written custom expressions and `ModelTerm` input at their boundaries
  (D365, #718, #568).


Ruling 10 deliberately changes the diagnostic for a static restart control on a
program of segments: construction rejects it before building any IR with
`a program of segments takes restart state through its activation, not a static restart control`.
The archived producer already failed verification with a non-dominating operand;
constructing that invalid SSA is not preserved. This is the sole exception to
exact diagnostic equivalence. The comparison records each affected case, target,
kind, stage and form with the complete old and new diagnostics, and accepts only
those exact pairs. Runtime continuation through activation between parts is
unchanged. Static restart support is tracked separately in #721.
