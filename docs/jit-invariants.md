# JIT memory and unwind ownership (D[jit-invariants])

Issue #89 strengthens D196 at the final host object boundary. No control-file
keys, file formats, defaults, or overwrite behavior change.

The simulation will own an ORC object memory manager. It will reserve a
contiguous allocation per object, inspect allocated executable sections and
relocated exception-frame records before registration, and reject layouts
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
failures from JIT failures. Numerical results will be compared with D196's
analytic pair oracle and the existing segment regressions.
