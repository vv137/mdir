# D[distribute-pme-observable-uses]

Recorded at its merge.

**Evaluate distributed PME observables according to their compiler uses.**

The lowering inspects the reciprocal energy and virial SSA results at each
mesh call. When both results are dead, the spectrum worker only multiplies
the complex coefficients by the influence factor. This skips the squared
magnitudes, observable scratch and MPI observable collective. An energy
consumer alone requests one sum. Any virial consumer requests the energy
and the nine virial components.

The requirement is constant for a call site and common to all ranks.
Synchronous and split calls select the same work for the same result uses.
The influence and gather arithmetic retain their precision and order.
Requested observables retain their per-plane and rank-ordered folds,
including the fixed-point word format.
