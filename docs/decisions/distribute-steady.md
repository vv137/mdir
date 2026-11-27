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
Mixed CPU midpoint rows subtract the explicit-image coordinates in double
before converting the displacement to kernel precision.

The native constrained CPU force loop retains a MATRIX layout, halo images
and list references across steps within half the pairlist skin. Membership
is keyed by particle identity within the numbering of the current loop.
The nondeterministic midpoint path reports the box of possible interaction
centers for its halo. A group member across a periodic face can be far
from that box, while its pair midpoint must lie inside. Group-local solves
require only owned rows.

Forward halo exchange groups peers by MPI shared-memory node. The intra-node
clock includes copies between ranks on that node, and the inter-node clock
covers exchange and unpack between nodes.
