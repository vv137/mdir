### Changed

- Add stream scopes, device field/table boundaries, mapped parameter and
  ordered contraction constructors, and the LJ correction constructor for
  the device-resident frame batch entry of #138 PR 1, D[frame-batch-entry].
  The C++ caller and private DLPack exercise entry write device energy,
  position gradients, strain and parameter rows; a torsion check covers
  batching and theta updates without restarting. General tunable integration
  and final validation remain in progress; existing frame evaluation is
  unchanged.
