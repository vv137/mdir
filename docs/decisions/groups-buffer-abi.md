# D[groups-buffer-abi]

Recorded at its merge.

**The compiler and runtime share one versioned neighbor-group buffer slot contract.**

The compiler and CUDA runtime include `mdir/Runtime/GroupsABI.h`. Explicit
slot numbers, element kinds, physical shapes, and roles are declared there.
The compiler selects slots by enum; the runtime checks the signed slot
before indexing its buffers and derives physical lengths from the descriptor.

The existing numbers, allocation sizes, tags, and generated IR are unchanged.
The group ABI version participates in the host compile-cache compatibility
key. Changing this contract requires increasing that version.

Part of #558; closes #623 when validated and merged. No control-file keys change.
