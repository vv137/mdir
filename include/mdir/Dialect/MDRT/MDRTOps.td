// Ops of the mdrt dialect.

#ifndef MDIR_DIALECT_MDRT_MDRTOPS_TD
#define MDIR_DIALECT_MDRT_MDRTOPS_TD

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mdir/Dialect/MD/MDTypes.td"
include "mdir/Dialect/MDRT/MDRTTypes.td"

class MDRT_Op<string mnemonic, list<Trait> traits = []>
    : Op<MDRT_Dialect, mnemonic, traits>;

def MDRT_FieldOrRelation
    : AnyTypeOf<[MD_FieldType, MD_RelationType, MD_TableType]>;

def MDRT_FromBufferOp : MDRT_Op<"from_buffer", [Pure]> {
  let summary = "The field or the tuples that a buffer holds";
  let description = [{
    Hands the contents of a buffer to the compiled program as a field. The
    program may overwrite the buffer afterward, so the buffer must not be
    read again except through `mdrt.to_buffer`.

    A field with 3 components of `f64` is held by a `memref<?x3xf64>`, and a
    field with 1 component by a `memref<?xT>`.

    ```mlir
    %x = mdrt.from_buffer %buffer
           : memref<?x3xf64> to !md.field<@atoms, 3 x f64>
    ```

    The buffer may hold `f32` where the field has `f64`. The field then has
    the values of the buffer, and the buffer states the type that the field
    is stored in. `md-exec-assign-precision` gives the field that type.

    ```mlir
    %x = mdrt.from_buffer %buffer
           : memref<?x3xf32> to !md.field<@atoms, 3 x f64>
    ```

    The tuples of a tuple set enter the program in the same way. The buffer
    holds, for each tuple, its members: the places of the particles in
    their fields. The program does not write to the buffer.

    ```mlir
    %angles = mdrt.from_buffer %members
                : memref<?x3xi32> to !md.relation<@atoms, 3, reversal, @angles>
    ```

    A table of rank `k` enters from a buffer of rank `k` with its element
    type, `memref<?x?xf64>` for a table of pairs of types.

    ```mlir
    %sigma = mdrt.from_buffer %buffer
               : memref<?x?xf64> to !md.table<2, f64, symmetric>
    ```
  }];
  let arguments = (ins AnyMemRef:$buffer);
  let results = (outs MDRT_FieldOrRelation:$result);
  let assemblyFormat = [{
    $buffer attr-dict `:` type($buffer) `to` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MDRT_ToBufferOp : MDRT_Op<"to_buffer", [Pure]> {
  let summary = "The buffer that holds a field";
  let description = [{
    The buffer is the one that holds the field when the op is reached. It
    need not be a buffer that was passed to `mdrt.from_buffer`.

    As for `mdrt.from_buffer`, the element type of the buffer states the
    type that the field is stored in.

    ```mlir
    %buffer = mdrt.to_buffer %x
                : !md.field<@atoms, 3 x f64> to memref<?x3xf64>
    ```
  }];
  let arguments = (ins MD_FieldType:$field);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $field attr-dict `:` qualified(type($field)) `to` type($result)
  }];
  let hasVerifier = 1;
}

def MDRT_HostCallOp : MDRT_Op<"host_call", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Calls a function of the host that reads fields";
  let description = [{
    Calls the function `callee`, which the host provides: to write a frame
    of a trajectory, for example. The function reads the fields that it is
    given while it runs and keeps nothing of them. The fields stay where
    they are, and the program goes on using them.

    `callee` is declared with a buffer where the op has a field. On a
    device, the field is copied to a buffer of the host for the call.

    ```mlir
    func.func private @write_frame(i64, memref<?x3xf32>)

    mdrt.host_call @write_frame(%step, %x)
        : (i64, !md.field<@atoms, 3 x f64>)
    ```

    As for `mdrt.from_buffer`, the element type of the buffer states the
    type that the field is stored in.

    With `in_place`, the function is given the buffers that hold the fields
    where they are, on a device as well: addresses of the device, which the
    host must not read as its own memory. The work before the call may
    still be under way on the device when it is made. A Python simulation
    takes its state this way at the end of each part (D[resident-buffers]).

    ```mlir
    mdrt.host_call @end_part(%x) {in_place} : (!md.field<@atoms, 3 x f64>)
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    UnitAttr:$in_place
  );
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict
    `:` `(` qualified(type($operands)) `)`
  }];
}

#endif // MDIR_DIALECT_MDRT_MDRTOPS_TD
