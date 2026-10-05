# JIT memory and unwind ownership (D[jit-invariants])

Issue #89 strengthens D196 at the final host object boundary. No control-file
keys, file formats, defaults, or overwrite behavior change.

The simulation owns an ORC object memory manager. It reserves a
contiguous allocation per object, inspect allocated executable sections and
relocated exception-frame records before registration, and rejects layouts
whose frame ranges leave their executable sections or whose bounding code
interval overlaps another live object. Section names alone are not evidence.
Late ORC initialization and deinitialization functions pass through this same
boundary. Registration is deferred until validation and memory finalization
succeed. Destruction deregisters frames before the allocation is released,
including failed construction. Creation, initialization, execution, and
teardown share the process runtime mutex.

The owned lifecycle is allocation/linking, validation, frame registration,
initialization, execution, deinitialization, deregistration, and release.
Validation failures publish no entry pointer. LLVM owns relocation correctness,
constructor/destructor synthesis, and object removal; libgcc owns exception
search and the implementation of its registration index. These assumptions
are checked with negative boundary tests and seeded lifecycle tests, rather
than claimed as formal verification of either dependency.

Validation will cover CPU/GPU, mixed/double, import orders, retained live
engines, construction errors, and delayed exceptions, with ASLR retained.
A separate host exception sanitizer baseline will distinguish interceptor
failures from JIT failures. Numerical results is compared with D196's
analytic pair oracle and the existing segment regressions.

## Enforced transitions and dependency boundary

Each object starts in `Linking`. RuntimeDyld allocates and relocates sections,
then requests frame registration. MDIR queues that request without calling the
unwinder. `finalizeMemory` checks actual allocated addresses, requires the
executable hull inside one owned mapping, requires exactly one unwind table
inside allocated data, parses its relocated CIE/FDE records with LLVM's DWARF
parser, and requires every FDE interval inside an executable section. Under the
process frame-registry mutex it rejects overlap with another live executable
hull. It applies permissions and only then registers the table and enters
`Registered`. Failure enters `Rejected` and registers nothing. A second
finalization or registration after release is rejected. Cleanup is idempotent:
it deregisters, removes the registry entry, and enters `Released`. The mapper
refuses storage release while the registration flag is still set.

The memory manager owns its mapper by base-class ordering: deregistration in
the derived destructor precedes SectionMemoryManager's release callbacks,
and the mapper outlives those callbacks. ORC removal also calls deregistration;
failed linking destroys the manager even when no resource was published.
The entry pointer is exposed only after initialization and successful lookup.
Initialization and symbol-resolution failures are returned as owned error
strings, which survive destruction of the ORC session.

Creation, continuation compilation, entry execution, and destruction take the
same process mutex. Polling Python happens between parts after releasing it.
A caller must keep a simulation alive while one of its methods runs; native
callers cannot destroy an object concurrently with its own active method.
The Python binding retains the object for its method call. Independent
simulations can remain live together. Their runtime operations serialize.

LLVM owns section sizing, overflow-safe reservation, relocation and machine
code correctness, frame encoding/decoding, permissions, constructor/destructor
synthesis and invocation, and object removal. MDIR verifies the resulting
layout rather than depending on section names or on a reservation promise.
LLVM's DWARF parser and libgcc's registration routines must agree on emitted
ELF frame records; MDIR does not validate arbitrary hostile object files.
The OS must keep simultaneously owned mappings disjoint. Contiguous ownership
also excludes unrelated libraries from the bounding interval, beyond the MDIR
registry. libgcc owns its index and deregistration implementation. CUDA owns
kernel module initialization and runtime cleanup. The supported host format
is ELF x86-64 or AArch64; other formats fail closed. Validation on this machine
covers x86-64; AArch64 is an unexercised dependency configuration.

The wrapper uses LLVM 23.1.2 ORC LLJIT and RuntimeDyld public interfaces.
Its packed entry matches the ABI documented by MLIR ExecutionEngine; the
implementation is independent and creates only the simulation's void wrapper.
No LLVM installation files are modified.
