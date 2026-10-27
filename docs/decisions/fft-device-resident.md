# D[fft-device-resident]

Recorded at its merge.

**Keep a bounded semantic FFT component resident on the device with checked layout, private backward inputs, and scoped host-boundary completion.** #580, part of #526 and #567. The design and validation are in [fft-device-resident.md](../fft-device-resident.md). Host remains the default; control keys, formats, checkpoints, and overwrite behavior are unchanged.

The first implementation shares packed scalar execution buffers rather than
native complex storage. Explicit scalar pack/unpack at semantic boundaries
is the layout-independent recovery. Placement rederives allocation, extent,
alias, queue, and release facts after bufferization and ownership analysis.
Unknown optional facts retain the host bridge with an explain remark;
contradictory IR is a diagnostic. D358 verifies the actual copy and blocking
completion before the scoped runtime call is selected. Inverse inputs are
private, and release waits for the actual producer queue.

The legacy-copy inventory and validation contract are in the design's
Sections 3 and 4. The host default remains unchanged; no measured result
licenses routing PME through this semantic path.
