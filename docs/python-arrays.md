# Host arrays for M2 (D[python-arrays])

Issue #78 replaces the D192 list boundary before persistent segments.
NumPy 1.23 or later is required when `MDIR_ENABLE_PYTHON=ON`; configuration
checks import and minimum version with the selected interpreter. CLI-only
builds retain no Python or NumPy requirement.

Inputs must export the buffer protocol or CPU DLPack. Coordinates have
shape `(N, 3)`, native float64 dtype and C-contiguous storage. Flat inputs,
lists, float32, nonnative byte order, nonfinite values and strided inputs
are refused with a property-specific `InputError`. Loaded states retain
their particle count; fresh states establish it on first nonempty assignment.
Absent velocities have shape `(0, 3)`.

Outputs are independent, C-contiguous, read-only NumPy copies: positions
and velocities `(N, 3)` float64, Cell vectors `(3, 3)` float64, Cell
diagonal/tilt `(3,)` float64, tuple particles `(n, arity)` int64 and each
named tuple parameter `(n,)` float64. Particle inputs accept int32 or int64,
with nonnegative values representable by native particle IDs. Parameter
collections retain their ordered name/array pairs. Configuration lists
such as water residue names and PME grid options retain their existing form.

Copy an output explicitly, edit it and assign it back. Assignment validates
and copies before advancing the state version. Failed assignment changes
neither values nor versions. Editing an independent returned copy, including
re-enabling its write flag, changes neither owner nor version.

CPU DLPack import copies into native ownership after dtype/shape/contiguity
validation; no gradients are retained. Non-CPU DLPack devices are refused.
Device views, leases and stream handoff remain the separate M2 DLPack gate.

No CLI keys, output/checkpoint formats or overwrite behavior change.
Validation covers buffer and CPU DLPack imports, malformed arrays, copies,
readonly flags, stale tracking, bitwise loader round trips, CPU/GPU
mixed/double IR parity, required-dependency configuration failures and the
full local suite. NumPy's [CPU DLPack importer](https://numpy.org/doc/1.23/reference/generated/numpy.from_dlpack.html)
and pybind11's [NumPy binding API](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)
provide the interchange machinery.
