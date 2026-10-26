// Ops of the dyn dialect.
//
// See docs/ops.md, Section 6, for the specification.

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
// Couplings (docs/dynamics.md, Section 2.2; D326)
//===----------------------------------------------------------------------===//

def Dyn_BathResource : Resource<"::mdir::dyn::BathResource">;
def Dyn_ChainResource : Resource<"::mdir::dyn::ChainResource">;
def Dyn_StopResource : Resource<"::mdir::dyn::StopResource">;

def Dyn_Vector3 : VectorOfLengthAndType<[3], [F64]>,
    BuildableType<"::mlir::VectorType::get({3}, $_builder.getF64Type())">;

// Measurements: global sums, which the host reads.

def Dyn_MomentumOp : Dyn_Op<"momentum", [
    Pure,
    TypesMatchWith<"the masses are a scalar field on the same particle set",
                   "velocities", "masses", Dyn_ScalarFieldOf>
  ]> {
  let summary = "The momentum of the particles";
  let description = [{
    ```text
    P = Σ_i m_i v_i
    ```

    One global sum. With `P / M` the velocity of the center of mass, which
    `dyn.shift` removes.

    ```mlir
    %p = dyn.momentum %v, %m : !md.field<@atoms, 3 x f64>
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$velocities,
    MD_ReferenceFieldType:$masses
  );
  let results = (outs Dyn_Vector3:$result);
  let assemblyFormat = [{
    $velocities `,` $masses attr-dict `:` qualified(type($velocities))
  }];
  let hasVerifier = 1;
}

def Dyn_KineticOp : Dyn_Op<"kinetic", [
    Pure,
    TypesMatchWith<"the masses are a scalar field on the same particle set",
                   "velocities", "masses", Dyn_ScalarFieldOf>
  ]> {
  let summary = "The kinetic energy of the particles";
  let description = [{
    ```text
    K_a = ½ Σ_i m_i v_{i,a}²     (vector<3xf64>, each axis)
    K   = Σ_i ½ (m_i (v_i · v_i)) (f64)
    ```

    One global sum. The two results are summed in different orders, and
    each is what the driver computes where it takes that result: the axes
    with a barostat, the total without one.

    ```mlir
    %k = dyn.kinetic %v, %m : !md.field<@atoms, 3 x f64> -> f64
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$velocities,
    MD_ReferenceFieldType:$masses
  );
  let results = (outs AnyTypeOf<[F64, Dyn_Vector3]>:$result);
  let assemblyFormat = [{
    $velocities `,` $masses attr-dict `:` qualified(type($velocities))
    `->` type($result)
  }];
  let hasVerifier = 1;
}

def Dyn_HalfStepExcessOp : Dyn_Op<"half_step_excess", [
    Pure,
    TypesMatchWith<"the masses are a scalar field on the same particle set",
                   "forces", "masses", Dyn_ScalarFieldOf>
  ]> {
  let summary = "The excess of the kinetic energy of the half steps";
  let description = [{
    ```text
    E_a = (Δt² / 8) Σ_i F_{i,a}² / m_i
    ```

    A particle of mass 0 is left out. Without constraints `K + E` is the
    mean of the kinetic energies of the two half steps around the step
    (Jung et al., J. Chem. Phys. 148, 164109 (2018)). One global sum.

    ```mlir
    %e = dyn.half_step_excess %f, %m, %dt : !md.field<@atoms, 3 x f64>
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$forces,
    MD_ReferenceFieldType:$masses,
    F64:$dt
  );
  let results = (outs Dyn_Vector3:$result);
  let assemblyFormat = [{
    $forces `,` $masses `,` $dt attr-dict `:` qualified(type($forces))
  }];
  let hasVerifier = 1;
}

// Draws and the thermostat with state: on the host.

def Dyn_VelocityRescalingOp : Dyn_Op<"velocity_rescaling", [Pure]> {
  let summary = "The factor of stochastic velocity rescaling";
  let description = [{
    The factor `α` of the velocities that brings the kinetic energy `K`
    to the next value of the stochastic process of Bussi, Donadio, and
    Parrinello, J. Chem. Phys. 126, 014101 (2007), its appendix, with the
    target `K̄ = ½ N_f k_B T_0`, `N_f` degrees of freedom, and the decay
    `exp(−N_T Δt / τ_T)` over the period; 1 if `K ≤ 0`. The numbers are
    drawn from Philox 4×32-10 under `seed`, at the step, in stream 0
    (docs/dynamics.md, Section 2.2.6): the op is a function of its
    operands and attributes.

    ```mlir
    %alpha = dyn.velocity_rescaling %k, %step
        {seed = 314159 : i64, target_kinetic = 1.1e+04, freedom = 8.9e+03,
         decay = 0.99} : f64
    ```
  }];
  let arguments = (ins
    F64:$kinetic,
    I64:$step,
    I64Attr:$seed,
    F64Attr:$target_kinetic,
    F64Attr:$freedom,
    F64Attr:$decay
  );
  let results = (outs F64:$result);
  let assemblyFormat = "$kinetic `,` $step attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Dyn_NoseHooverOp : Dyn_Op<"nose_hoover", [
    MemoryEffects<[MemRead<Dyn_ChainResource>, MemWrite<Dyn_ChainResource>,
                   MemWrite<Dyn_StopResource>]>
  ]> {
  let summary = "The factor of a Nose-Hoover chain over a period";
  let description = [{
    Moves the chain of `chain_length` thermostats over the period by the
    kinetic energy `K`, and returns the factor of the velocities (Martyna,
    Klein, and Tuckerman, J. Chem. Phys. 97, 2635 (1992); Martyna et al.,
    Mol. Phys. 87, 1117 (1996); D163a). The chain is a state of the host,
    which the checkpoint keeps. The step names the step in the message of
    a stop, when the factorization no longer follows the chain (D206).

    ```mlir
    %alpha = dyn.nose_hoover %k, %step
        {temperature = 300.0, time_constant = 1.0, chain_length = 3 : i64,
         period = 0.02, freedom = 8.9e+03} : f64
    ```
  }];
  let arguments = (ins
    F64:$kinetic,
    I64:$step,
    F64Attr:$temperature,
    F64Attr:$time_constant,
    I64Attr:$chain_length,
    F64Attr:$period,
    F64Attr:$freedom
  );
  let results = (outs F64:$result);
  let assemblyFormat = "$kinetic `,` $step attr-dict `:` type($result)";
  let hasVerifier = 1;
}

// Updates of the particles.

def Dyn_ShiftOp : Dyn_Op<"shift", [
    Pure, AllTypesMatch<["velocities", "result"]>
  ]> {
  let summary = "Subtracts a vector from every velocity";
  let description = [{
    ```text
    v'_i = v_i − u
    ```

    With `u = P / M` the removal of the motion of the center of mass.

    ```mlir
    %v1 = dyn.shift %v, %u : !md.field<@atoms, 3 x f64>
    ```
  }];
  let arguments = (ins MD_ReferenceFieldType:$velocities, Dyn_Vector3:$shift);
  let results = (outs MD_ReferenceFieldType:$result);
  let assemblyFormat = [{
    $velocities `,` $shift attr-dict `:` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Dyn_ScaleOp : Dyn_Op<"scale", [
    Pure, AllTypesMatch<["velocities", "result"]>
  ]> {
  let summary = "Scales every velocity";
  let description = [{
    ```text
    v'_i = s v_i      (f64)
    v'_i = s ⊙ v_i    (vector<3xf64>)
    ```

    ```mlir
    %v1 = dyn.scale %v, %alpha : !md.field<@atoms, 3 x f64>, f64
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$velocities,
    AnyTypeOf<[F64, Dyn_Vector3]>:$factor
  );
  let results = (outs MD_ReferenceFieldType:$result);
  let assemblyFormat = [{
    $velocities `,` $factor attr-dict `:` qualified(type($result)) `,`
    type($factor)
  }];
  let hasVerifier = 1;
}

def Dyn_NoiseOp : Dyn_Op<"noise", [
    Pure,
    AllTypesMatch<["velocities", "result"]>,
    TypesMatchWith<"the masses are a scalar field on the same particle set",
                   "velocities", "masses", Dyn_ScalarFieldOf>
  ]> {
  let summary = "Damps the velocities and adds normal numbers";
  let description = [{
    ```text
    v'_i = c v_i + (σ / √m_i) R_i
    ```

    `R_i` are three standard normal numbers of particle `i` at the step:
    one block of Philox 4×32-10 under `seed`, with the counter of the step,
    the number of the particle in the input (`ids`), and stream 2
    (docs/dynamics.md, Section 2.2.6). A particle of mass 0 keeps its
    velocity. With `c = exp(−γ Δt)` and `σ = √((1 − c²) k_B T_0)` the O
    step of Langevin dynamics; with `c = 1` and `σ = √(2 k_B T_0 / (γ Δt))`
    on the velocity of the drift, the noise of Brownian dynamics.

    ```mlir
    %v1 = dyn.noise %v, %m, %step, %ids
        {seed = 314159 : i64, decay = 0.99, spread = 0.22}
        : !md.field<@atoms, 3 x f64>, !md.field<@atoms, i32>
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$velocities,
    MD_ReferenceFieldType:$masses,
    I64:$step,
    MD_ReferenceFieldType:$ids,
    I64Attr:$seed,
    F64Attr:$decay,
    F64Attr:$spread
  );
  let results = (outs MD_ReferenceFieldType:$result);
  let assemblyFormat = [{
    $velocities `,` $masses `,` $step `,` $ids attr-dict `:`
    qualified(type($velocities)) `,` qualified(type($ids))
  }];
  let hasVerifier = 1;
}

// The state of the host.

def Dyn_ToBathOp : Dyn_Op<"to_bath", [
    MemoryEffects<[MemRead<Dyn_BathResource>, MemWrite<Dyn_BathResource>]>
  ]> {
  let summary = "Adds an energy to the bath";
  let description = [{
    Adds `e`, the energy that a coupling took from the system, to the
    energy of the bath, which the conserved energy of the log and the
    checkpoint's `bath` take (D129).

    ```mlir
    dyn.to_bath %heat : f64
    ```
  }];
  let arguments = (ins F64:$energy);
  let assemblyFormat = "$energy attr-dict `:` type($energy)";
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
    `barostatting`, `no_global_sums`.

    The verifier checks the claims against the body (docs/dynamics.md,
    Section 2.2.4). A claim stays true when `dyn-expand-couplings` expands
    the ops it rests on, so a rule holds in the direction from the body to
    the claim:

    | Name | Rule |
    |---|---|
    | `requires = ["temperature"]`, `provides = ["thermostatting"]` | If the body has a `dyn.noise` |
    | `provides = ["symplectic"]`, `["time_reversible"]` | Only if the body has no `dyn.noise` |
    | `provides = ["no_global_sums"]` | Only if the body computes no global sum: no measurement of `dyn`, no `md.sum_particles`, `md.sum_tuples`, or `md.sum_relation` of a floating-point result, and no `md.evaluate` of the energy, the virial, or a derivative |

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
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Applies a program";
  let description = [{
    A step has the effects of the ops of the program it applies, on the
    resources those ops name: a program that calls the host, or writes a
    state of the host, is not moved across another step or removed (docs/
    dynamics.md, Section 2.2.3). An op of unknown effects, such as a
    `func.call`, reads and writes every resource.

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
