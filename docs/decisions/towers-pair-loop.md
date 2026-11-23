# D[towers-pair-loop]

Recorded at its merge.

**Stage small packed type-pair coefficient tables in each TOWERS block.**
The moving traversal specializes the shared table dimensions to the topology's
type count when the pair region indexes the packed coefficients by particle
type. Threads copy the entries cooperatively before traversal. The copy
respects both dimensions of the source descriptor. The pair loop indexes
the shared table with a fixed row stride. The existing table-size and
static shared-memory limits bound the specialization.
