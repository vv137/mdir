# D[builder-native-program]

Recorded at its merge.

**The builder constructs a program from owned native IR; only user input is parsed at its boundary.**

This is stage C of #568, after the complete term adapter registry and D363's
typed term fragments. The construction contract and current status belong in
[driver Section 2.12](../driver.md#212-the-files-of-the-builder).

## Ownership and construction

One build owns its MLIR context, module, parsed model modules, and construction
captures. Input IR and destination IR share that context. A term borrows typed
operands and an `OpBuilder` insertion point through a bounded capability;
neither a mutable `Program` nor the complete `Builder` crosses the adapter
boundary. Borrowed state does not escape a hook.

Custom expressions retain their owned `ExpressionIR` snapshots. Staged
`ModelTerm` input is parsed once, with its potential signature and return
checked at that boundary. Declarations and the potential body are cloned with
`IRMapping`, preserving declaration and operation order. Symbols supplied by
the complete program, such as `@atoms`, are checked at the ordinary combined
verification boundary. Duplicate user declarations are not silently renamed.
There is no substring search or SSA-token rewriting of model input.

Program declarations, shared potential wrappers, constraints, dynamics,
entry functions, schedules, segments, and output calls are constructed with
`OpBuilder`. Their interfaces take and return typed values and operations.
Generated operation or type syntax is not formatted and reparsed, and an
SSA-name dictionary is not a substitute for those typed interfaces. Once the
last caller moves, connection serialization and carry-marker fixups disappear.

The completed native module is printed for the existing serialized
`Program.module` handoff. Ordinary consumer verification stays at its existing
boundary. Contribution and field order, shared potential and loop placement,
inlining, fusion, and MLIR's default SSA numbering remain unchanged; builder
name hints and new opaque call boundaries are not introduced.

## Validation and status

Control keys, defaults, support restrictions, numerical methods, precision
modes, formats, and overwrite behavior are unchanged. Every implementation
commit compares the full printed module/lowered corpus and combined order
matrix against one archived PR base. The final head compares five checkpoint
states bitwise and passes both clean D309 shards.

Staged-model input parsing and cloning are implemented in `StagedModelIR.cpp`.
The build context outlives the adapters and their captured input modules;
recipe construction errors propagate to the producer. Input buffers remain
owned for parsed resource lifetimes. Native tests cover the frozen exported
model, user string literals and nondefault SSA names, signature rejection,
foreign program operands, and context ownership. The remaining program
construction follows. D363's term recipes still join generated assembly at
this commit.
