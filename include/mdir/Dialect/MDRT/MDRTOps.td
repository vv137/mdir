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

def MDRT_HaloCopyStartOp : MDRT_Op<"halo_copy_start"> {
  let summary = "Starts an owner-to-ghost buffer transfer";
  let description = [{
    Starts the physical transfer of the existing `md_dist.halo_copy` (G).
    `source` and `destination` remain live through `mdrt.halo_wait`.
    `image` is one for a position copy with periodic images, zero otherwise.
    The event has exactly one same-block wait. The reference lowering calls
    the synchronous in-process transport at this op.
  }];
  let arguments = (ins I64:$halo, AnyMemRef:$source,
                       AnyMemRef:$destination, I64:$image);
  let results = (outs MDRT_EventType:$event);
  let assemblyFormat = [{
    $halo `,` $source `,` $destination `,` $image attr-dict
    `:` type($source) `,` type($destination)
  }];
  let hasVerifier = 1;
}

def MDRT_HaloSumStartOp : MDRT_Op<"halo_sum_start"> {
  let summary = "Starts a ghost-to-owner buffer sum";
  let description = [{
    Starts the physical transfer of the existing `md_dist.halo_sum`
    (G^T). The destination already holds the owned contribution.
    Its event and buffers have the same completion contract as
    `mdrt.halo_copy_start`.
  }];
  let arguments = (ins I64:$halo, AnyMemRef:$source,
                       AnyMemRef:$destination);
  let results = (outs MDRT_EventType:$event);
  let assemblyFormat = [{
    $halo `,` $source `,` $destination attr-dict
    `:` type($source) `,` type($destination)
  }];
  let hasVerifier = 1;
}

def MDRT_HaloWaitOp : MDRT_Op<"halo_wait"> {
  let summary = "Completes a halo transfer and yields its destination";
  let description = [{
    Waits for one halo start event, retaining its exact halo handle and
    source/destination buffers until completion. The result is the
    destination as a completed SSA buffer version; consumers of the
    transferred value use this result.
  }];
  let arguments = (ins MDRT_EventType:$event, I64:$halo,
                       AnyMemRef:$source, AnyMemRef:$destination);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $event `,` $halo `,` $source `,` $destination attr-dict
    `:` type($source) `,` type($destination) `->` type($result)
  }];
  let hasVerifier = 1;
}

def MDRT_HostCallOp : MDRT_Op<"host_call", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
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
    takes its state this way at the end of each part (D215).

    ```mlir
    mdrt.host_call @end_part(%x) {in_place} : (!md.field<@atoms, 3 x f64>)
    ```

    A function that writes fields takes them in `outs`, as the loops of
    md_exec take their destinations: it adds to what the buffer of each
    holds, and the op yields the field that the buffer holds after the
    call, one result for each field in `outs`. A loop that reads these
    forces reads the result, so no pass can move it, or fuse it, before
    the call. A field of `outs` must be dead after the call; its buffer
    becomes that of the result. The numbers that the function computes
    (`f64`, or a vector of `f64`) follow the fields among the results: the
    function writes each to a buffer of the host (`memref<1xf64>`, or
    `memref<kxf64>`) that it is given after those of the fields, and the
    host waits for the call. A learned potential adds its forces to those
    of the other terms this way, and gives its energy and its virial
    (D270).

    ```mlir
    func.func private @model(memref<?x3xf64>, memref<?x3xf32>,
                             memref<1xf64>, memref<9xf64>)

    %f, %u, %w = mdrt.host_call @model(%x) outs(%fc : !vec) {in_place}
        : (!vec) -> (!vec, f64, vector<9xf64>)
    ```

    The op reads and writes the state of the host, which the compiler does
    not see: it is never removed or reordered with other such ops.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$inputs,
    Variadic<MD_FieldType>:$outs,
    UnitAttr:$in_place
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $inputs `)`
    (`outs` `(` $outs^ `:` qualified(type($outs)) `)`)? attr-dict
    `:` `(` qualified(type($inputs)) `)`
    (`->` `(` qualified(type($results))^ `)`)?
  }];
  let extraClassDeclaration = [{
    /// The numbers among the results, after the fields of `outs`.
    ::mlir::ResultRange getNumbers() {
      return getResults().drop_front(getOuts().size());
    }
  }];
  let hasVerifier = 1;
}

#endif // MDIR_DIALECT_MDRT_MDRTOPS_TD
