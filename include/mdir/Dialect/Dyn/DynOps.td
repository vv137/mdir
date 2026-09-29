// Ops of the dyn dialect.
//
// See docs/ops-m0.md, Section 6, for the specification.

#ifndef MDIR_DIALECT_DYN_DYNOPS_TD
#define MDIR_DIALECT_DYN_DYNOPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mdir/Dialect/Dyn/DynDialect.td"
include "mdir/Dialect/MD/MDTypes.td"

// The type of the scalar field on the particle set of the vector field
// `$_self`.
defvar Dyn_ScalarFieldOf =
    "::mdir::md::FieldType::get($_self.getContext(), "
    "::llvm::cast<::mdir::md::FieldType>($_self).getParticleSet(), 1, "
    "::llvm::cast<::mdir::md::FieldType>($_self).getElementType())";

//===----------------------------------------------------------------------===//
// Updates
//===----------------------------------------------------------------------===//

def Dyn_KickOp : Dyn_Op<"kick", [
    Pure,
    AllTypesMatch<["velocities", "forces", "result"]>,
    TypesMatchWith<"the masses are a scalar field on the same particle set",
                   "result", "masses", Dyn_ScalarFieldOf>
  ]> {
  let summary = "Advances the velocities under a force";
  let description = [{
    ```text
    v'_i = v_i + dt · f_i / m_i
    ```

    A particle of mass 0, a virtual site, keeps its velocity.

    ```mlir
    %v1 = dyn.kick %v, %f, %m, %dt : !md.field<@atoms, 3 x f64>
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$velocities,
    MD_ReferenceFieldType:$forces,
    MD_ReferenceFieldType:$masses,
    F64:$dt
  );
  let results = (outs MD_ReferenceFieldType:$result);
  let assemblyFormat = [{
    $velocities `,` $forces `,` $masses `,` $dt attr-dict
    `:` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Dyn_DriftOp : Dyn_Op<"drift", [
    Pure,
    AllTypesMatch<["positions", "velocities", "result"]>
  ]> {
  let summary = "Advances the positions at constant velocity";
  let description = [{
    ```text
    x'_i = x_i + dt · v_i
    ```

    The op does not wrap positions into the cell. Whether stored positions
    are wrapped is a storage decision.

    ```mlir
    %x1 = dyn.drift %x, %v, %dt : !md.field<@atoms, 3 x f64>
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$positions,
    MD_ReferenceFieldType:$velocities,
    F64:$dt
  );
  let results = (outs MD_ReferenceFieldType:$result);
  let assemblyFormat = [{
    $positions `,` $velocities `,` $dt attr-dict `:` qualified(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Programs
//===----------------------------------------------------------------------===//

def Dyn_ProgramOp : Dyn_Op<"program", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "Defines one time step";
  let description = [{
    A program takes the state and the parameters and returns the new state.

    | Attribute | Meaning |
    |---|---|
    | `velocity_offset` | Time of the stored velocities relative to the positions, in units of the time step. 0 if absent. |
    | `requires` | What the program needs from the thermodynamic state. |
    | `provides` | Properties of the program. |

    Known requirements: `temperature`, `pressure`.

    Known properties: `symplectic`, `time_reversible`, `thermostatting`,
    `barostatting`.

    ```mlir
    dyn.program @leapfrog(%x: !vec, %v: !vec, %m: !real, %cell: !md.cell,
                          %dt: f64) -> (!vec, !vec)
        attributes {velocity_offset = -0.5,
                    provides = ["symplectic", "time_reversible"]} {
      ...
      dyn.return %x1, %v1 : !vec, !vec
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<F64Attr>:$velocity_offset,
    OptionalAttr<StrArrayAttr>:$requires,
    OptionalAttr<StrArrayAttr>:$provides
  );
  let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }
    bool isDeclaration() { return isExternal(); }

    /// The time of the stored velocities relative to the positions, in
    /// units of the time step.
    double getVelocityOffsetOrZero() {
      auto offset = getVelocityOffset();
      return offset ? offset->convertToDouble() : 0.0;
    }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Dyn_ReturnOp : Dyn_Op<"return", [
    Pure, Terminator, ReturnLike, HasParent<"ProgramOp">
  ]> {
  let summary = "Returns the new state from a program";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let hasVerifier = 1;
}

def Dyn_StepOp : Dyn_Op<"step", [
    Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Applies a program";
  let description = [{
    ```mlir
    %x1, %v1 = dyn.step @leapfrog(%x, %v, %m, %cell, %dt)
                 : (!vec, !vec, !real, !md.cell, f64) -> (!vec, !vec)
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict
    `:` functional-type($operands, results)
  }];
}

#endif // MDIR_DIALECT_DYN_DYNOPS_TD
