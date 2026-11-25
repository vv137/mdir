# D[distribute-steady]

Recorded at its merge.

Distributed CPU step timing separates MATRIX builds, halo membership,
forward MPI packing, local copies, remote exchanges, state handling,
integration and writer calls. Nested phase times exclude their children.
Owned field capacity follows the largest domain at each ownership epoch.
The MPI transport caches its row schedules until the halo is released.
Reverse reception visits contributions in domain and ghost order,
including local entries, to preserve the order of force accumulation.
Validated constraint groups read every tuple member at its owner.
