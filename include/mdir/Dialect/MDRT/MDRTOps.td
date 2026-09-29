// Ops of the mdrt dialect.

#ifndef MDIR_DIALECT_MDRT_MDRTOPS_TD
#define MDIR_DIALECT_MDRT_MDRTOPS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mdir/Dialect/MD/MDTypes.td"
include "mdir/Dialect/MDRT/MDRTTypes.td"

class MDRT_Op<string mnemonic, list<Trait> traits = []>
    : Op<MDRT_Dialect, mnemonic, traits>;

def MDRT_FromBufferOp : MDRT_Op<"from_buffer", [Pure]> {
  let summary = "The field that a buffer holds";
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
  }];
  let arguments = (ins AnyMemRef:$buffer);
  let results = (outs MD_FieldType:$result);
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

#endif // MDIR_DIALECT_MDRT_MDRTOPS_TD
