# D[bonded-workunits]

Recorded at its merge.

Evaluate atom-local valence work units in an offline experiment. Build
`bonded-partition` and give it the same control file as the simulation.
A term is repeated once per distinct owning interval it touches. Each
member's owner takes its force, and the first member's owner takes the
term energy and virial. The experiment checks the packed local indices
and both ownership rules for each term. The companion target
`bonded-device-check` verifies shared position staging and signed integer
payloads against a direct CPU oracle with zero tolerance.

At 128 atoms per interval the extra valence evaluations range from 5.62%
to 6.55% on the four benchmark systems. A shared integer force accumulator
would require a force scale and a policy for overflow, as well as a
numerical contract for comparisons with floating-point tuple forces.
The mixed kernel on the device sums in floating point. Deterministic
evaluation walks ordered rows. [Partition and costs](../device-tuples.md#12-topology-order-work-units)
