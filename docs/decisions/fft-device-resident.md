# D[fft-device-resident]

Recorded at its merge.

**Keep a bounded semantic FFT component resident on the device with checked layout, private backward inputs, and scoped host-boundary completion.** #580, part of #526 and #567. The design and validation are in [fft-device-resident.md](../fft-device-resident.md). Host remains the default; control keys, formats, checkpoints, and overwrite behavior are unchanged.
