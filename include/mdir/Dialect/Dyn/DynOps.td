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
def Dyn_RecordResource : Resource<"::mdir::dyn::RecordResource">;
def Dyn_StatisticsResource : Resource<"::mdir::dyn::StatisticsResource">;

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

// The barostat (step 2b of #541).

def Dyn_CellMemory : MemRefRankOf<[F64], [1]>;

def Dyn_GroupKineticOp : Dyn_Op<"group_kinetic", [
    Pure, AttrSizedOperandSegments,
    AllTypesMatch<["positions", "velocities", "forces"]>,
    TypesMatchWith<"the masses are a scalar field on the same particle set",
                   "velocities", "masses", Dyn_ScalarFieldOf>
  ]> {
  let summary = "The kinetic energy of the centers of the groups";
  let description = [{
    The groups are those that a barostat scales by their centers: each
    rigid group of the relations in `groups`, and each other particle
    alone (D264). The results, axis by axis:

    ```text
    K_g      = Σ_g ½ M_g V_g²            (less that of the center of mass,
                                           with `center`, the total mass)
    E_g      = (Δt² / 8) Σ_g F_g² / M_g
    K_half_g = K_g + E_g
    ```

    `K_half_g` is the mean of the kinetic energies of the centers at the
    half steps around the step, exactly (Jung and Sugita, J. Chem. Phys. 153,
    234115 (2020), eq. 42): a center moves by its force alone. Global sums
    over the particles and over the tuples of each relation.

    ```mlir
    %k, %e, %h = dyn.group_kinetic %x, %v, %f, %m, %cell, %dt
        groups(%w : !md.relation<@atoms, 3, ordered, @settles>)
        center(%total) : !md.field<@atoms, 3 x f64>
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$positions,
    MD_ReferenceFieldType:$velocities,
    MD_ReferenceFieldType:$forces,
    MD_ReferenceFieldType:$masses,
    MD_CellType:$cell,
    F64:$dt,
    Variadic<MD_RelationType>:$groups,
    Optional<Dyn_Vector3>:$center
  );
  let results = (outs Dyn_Vector3:$kinetic, Dyn_Vector3:$excess,
                      Dyn_Vector3:$half);
  let assemblyFormat = [{
    $positions `,` $velocities `,` $forces `,` $masses `,` $cell `,` $dt
    (`groups` `(` $groups^ `:` qualified(type($groups)) `)`)?
    (`center` `(` $center^ `)`)? attr-dict `:` qualified(type($velocities))
  }];
  let hasVerifier = 1;
}

def Dyn_PressureOp : Dyn_Op<"pressure", [Pure]> {
  let summary = "The pressure of each axis";
  let description = [{
    ```text
    P_aa = c_bar (2 K_a + W_aa + C / (3 V)) / V,   V = (L_0 L_1) L_2
    ```

    In bar, with `K` the kinetic energy of the groups at the half steps,
    `W` the diagonal of their virial, `C` the virials of the constant terms
    times the volume, and c_bar = 16.6053906717 bar nm³ mol/kJ (D264).

    ```mlir
    %p = dyn.pressure %k, %w, %l0, %l1, %l2, %c : vector<3xf64>
    ```
  }];
  let arguments = (ins Dyn_Vector3:$kinetic, Dyn_Vector3:$virial, F64:$l0,
                       F64:$l1, F64:$l2, F64:$constant);
  let results = (outs Dyn_Vector3:$result);
  let assemblyFormat = [{
    $kinetic `,` $virial `,` $l0 `,` $l1 `,` $l2 `,` $constant attr-dict
    `:` type($result)
  }];
}

def Dyn_CellRescalingOp : Dyn_Op<"cell_rescaling", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "The change of the strain of stochastic cell rescaling";
  let description = [{
    The change Δε_a of the strain of each axis over the period, from the
    pressures `p` of the axes in bar (Bernetti and Bussi, J. Chem. Phys.
    153, 114107 (2020)):

    | `coupling` | Δε |
    |---|---|
    | `isotropic` | A third of Δε of the step of λ = √V, eq. (S7), from the mean of the three pressures, on every axis |
    | `semi_isotropic` | Half of Δε_xy, eq. (9a), on x and y, and Δε_z, eq. (9b), on z (D119) |
    | `anisotropic` | Δε_a of each axis from its own pressure (D163c) |

    `compressibility` holds one number, two (of x and y, and of z), or
    three, in 1/bar; `rate` is N_P Δt / τ_P, `kT` in kJ/mol, and
    `tension` γ in bar nm, semi-isotropic only. The noise is drawn from
    Philox 4×32-10 under `seed`, at the step, in stream 1
    (docs/dynamics.md, Section 2.2.6). Semi-isotropic and anisotropic
    coupling also write the statistics of the pressures of the axes that
    the log reports at the end of the run.

    ```mlir
    %de = dyn.cell_rescaling %p, %l0, %l1, %l2, %step
        {coupling = "isotropic", seed = 314159 : i64, target = 1.01325,
         compressibility = [4.5e-05], rate = 0.01, kT = 2.494} : vector<3xf64>
    ```
  }];
  let arguments = (ins
    Dyn_Vector3:$pressures,
    F64:$l0, F64:$l1, F64:$l2,
    I64:$step,
    StrAttr:$coupling,
    I64Attr:$seed,
    F64Attr:$target,
    F64ArrayAttr:$compressibility,
    F64Attr:$rate,
    F64Attr:$kT,
    OptionalAttr<F64Attr>:$tension
  );
  let results = (outs Dyn_Vector3:$result);
  let assemblyFormat = [{
    $pressures `,` $l0 `,` $l1 `,` $l2 `,` $step attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def Dyn_ScaleCellOp : Dyn_Op<"scale_cell", [Pure]> {
  let summary = "Scales the cell by the factor of each axis";
  let description = [{
    `H diag(μ)`: the edges by their μ_a and, of a triclinic cell, the tilts
    b_x and c_x by μ_x and c_y by μ_y (D127). Three or six values, as many
    as the results.

    ```mlir
    %n:3 = dyn.scale_cell %l0, %l1, %l2 by %mu : f64, f64, f64 -> f64, f64, f64
    ```
  }];
  let arguments = (ins Variadic<F64>:$cell, Dyn_Vector3:$factors);
  let results = (outs Variadic<F64>:$results);
  let assemblyFormat = [{
    $cell `by` $factors attr-dict `:` type($cell) `->` type($results)
  }];
  let builders = [OpBuilder<(ins "::mlir::ValueRange":$cell,
                                 "::mlir::Value":$factors), [{
    build($_builder, $_state,
          ::llvm::SmallVector<::mlir::Type>(cell.size(),
                                            $_builder.getF64Type()),
          cell, factors);
  }]>];
  let hasVerifier = 1;
}

def Dyn_LoadCellOp : Dyn_Op<"load_cell"> {
  let summary = "Reads the cell";
  let description = [{
    The edges, and of a triclinic cell the tilts b_x, c_x, c_y, from the
    memory of the cell, a memref of 3 or 6 f64 in any memory space (D72):
    on the host today, on a device for a barostat there.

    ```mlir
    %e:3 = dyn.load_cell %box : memref<3xf64> -> f64, f64, f64
    ```
  }];
  let arguments = (ins Arg<Dyn_CellMemory, "the cell", [MemRead]>:$memory);
  let results = (outs Variadic<F64>:$cell);
  let assemblyFormat = [{
    $memory attr-dict `:` type($memory) `->` type($cell)
  }];
  let hasVerifier = 1;
}

def Dyn_StoreCellOp : Dyn_Op<"store_cell", [
    MemoryEffects<[MemWrite<Dyn_RecordResource>, MemWrite<Dyn_StopResource>]>
  ]> {
  let summary = "Writes the cell";
  let description = [{
    Writes the edges, and of a triclinic cell the tilts, into the memory of
    the cell, and gives them to the log, the trajectory, and the checkpoint.
    A cell that is not finite, or that one coupling has scaled by more than
    a factor of 2 along an axis, stops the run (#168, D225).

    ```mlir
    dyn.store_cell %n#0, %n#1, %n#2 into %box : memref<3xf64>
    ```
  }];
  let arguments = (ins Variadic<F64>:$cell,
                       Arg<Dyn_CellMemory, "the cell", [MemWrite]>:$memory);
  let assemblyFormat = [{
    $cell `into` $memory attr-dict `:` type($memory)
  }];
  let hasVerifier = 1;
}

def Dyn_ScalePositionsOp : Dyn_Op<"scale_positions", [
    Pure, AllTypesMatch<["positions", "result"]>,
    TypesMatchWith<"the masses are a scalar field on the same particle set",
                   "positions", "masses", Dyn_ScalarFieldOf>
  ]> {
  let summary = "Scales the positions by the factor of each axis";
  let description = [{
    `μ ⊙ x_i` for a particle alone; each rigid group of the relations in
    `groups` moves with its center of mass X_g to μ ⊙ X_g and keeps its
    shape (D72): a particle of a group moves back by (μ − 1) times its
    place about the center.

    ```mlir
    %x1 = dyn.scale_positions %x, %mu, %m, %cell
        groups(%w : !md.relation<@atoms, 3, ordered, @settles>)
        : !md.field<@atoms, 3 x f64>
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$positions,
    Dyn_Vector3:$factors,
    MD_ReferenceFieldType:$masses,
    MD_CellType:$cell,
    Variadic<MD_RelationType>:$groups
  );
  let results = (outs MD_ReferenceFieldType:$result);
  let assemblyFormat = [{
    $positions `,` $factors `,` $masses `,` $cell
    (`groups` `(` $groups^ `:` qualified(type($groups)) `)`)?
    attr-dict `:` qualified(type($positions))
  }];
  let hasVerifier = 1;
}

def Dyn_ScalingWorkOp : Dyn_Op<"scaling_work", [
    Pure, AttrSizedOperandSegments
  ]> {
  let summary = "The energy that a scaling of the cell gives the system";
  let description = [{
    ΔE_sys, of the velocities exactly, Σ_a (μ_a⁻² − 1) K_a, and of the
    positions as `count` says (docs/dynamics.md, Section 2.2.7), with `C`
    the constant terms and V, V' the volumes before and after:

    | `count` | Operands | Positions |
    |---|---|---|
    | `trotter` | `log_strain`, `virial_before`, `virial_after`, `volumes(V, V')` | −Σ_a ln μ_a ½ (W_b,aa + C/(3V) + W_a,aa + C/(3V')) |
    | `trotter_before` | `log_strain`, `virial_before`, `volumes(V)` | −Σ_a ln μ_a (W_b,aa + C/(3V)) |
    | `exact` | `strain`, `energies(U, U')`, `volumes(V, V')` | U' − U + C (1/V' − 1/V) |
    | `first_order` | `strain`, `virial_before`, `volumes(V)` | −Σ_a (μ_a − 1) ⅓ (tr W + C/V) |

    `kinetic` is that of the velocities that the scaling scales; with
    `log_strain` μ = exp(ln μ). `constant` is C: the virials of the
    constant terms times the volume, or with `exact` their energy times it.

    ```mlir
    %w = dyn.scaling_work kinetic(%k) strain(%mu) virial_before(%wg)
        constant(%c) volumes(%v) {count = "first_order"} : f64
    ```
  }];
  let arguments = (ins
    Dyn_Vector3:$kinetic,
    Optional<Dyn_Vector3>:$strain,
    Optional<Dyn_Vector3>:$log_strain,
    Optional<Dyn_Vector3>:$virial_before,
    Optional<Dyn_Vector3>:$virial_after,
    Variadic<F64>:$energies,
    F64:$constant,
    Variadic<F64>:$volumes,
    StrAttr:$count
  );
  let results = (outs F64:$result);
  let assemblyFormat = [{
    `kinetic` `(` $kinetic `)` (`strain` `(` $strain^ `)`)?
    (`log_strain` `(` $log_strain^ `)`)?
    (`virial_before` `(` $virial_before^ `)`)?
    (`virial_after` `(` $virial_after^ `)`)?
    (`energies` `(` $energies^ `)`)? `constant` `(` $constant `)`
    `volumes` `(` $volumes `)` attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

// The state of the pressure of a barostat of Trotter type that scales
// every step (step 2c of #541; D92).

def Dyn_StateMemory : MemRefRankOf<[F64], [1]>;

def Dyn_LoadPressureStateOp : Dyn_Op<"load_pressure_state"> {
  let summary = "Reads the state of the pressure that the step before left";
  let description = [{
    The diagonal of the virial, that of the virial of the groups, and the
    kinetic energy of the centers of the groups at the half steps after the
    thermostat, of the step before: the pressure of a barostat of Trotter
    type with a period of one step (D92). The memory is a memref of 9 f64 in
    any memory space.

    ```mlir
    %w, %wg, %k = dyn.load_pressure_state %state : memref<9xf64>
    ```
  }];
  let arguments = (ins Arg<Dyn_StateMemory, "the state", [MemRead]>:$memory);
  let results = (outs Dyn_Vector3:$virial, Dyn_Vector3:$groups,
                      Dyn_Vector3:$kinetic);
  let assemblyFormat = "$memory attr-dict `:` type($memory)";
  let hasVerifier = 1;
}

def Dyn_StorePressureStateOp : Dyn_Op<"store_pressure_state", [
    MemoryEffects<[MemWrite<Dyn_RecordResource>]>
  ]> {
  let summary = "Writes the state of the pressure for the next step";
  let description = [{
    Writes the state that `dyn.load_pressure_state` reads into the memory,
    and gives it to the checkpoint (`/parameters/mdir/barostat_state`), so
    that a run that continues takes the same.

    ```mlir
    dyn.store_pressure_state %w, %wg, %k into %state : memref<9xf64>
    ```
  }];
  let arguments = (ins Dyn_Vector3:$virial, Dyn_Vector3:$groups,
                       Dyn_Vector3:$kinetic,
                       Arg<Dyn_StateMemory, "the state", [MemWrite]>:$memory);
  let assemblyFormat = [{
    $virial `,` $groups `,` $kinetic `into` $memory attr-dict `:`
    type($memory)
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
    FunctionOpInterface, IsolatedFromAbove,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Defines one time step";
  let description = [{
    A program takes the state and the parameters and returns the new state.

    | Attribute | Meaning |
    |---|---|
    | `velocity_offset` | Time of the stored velocities relative to the positions, in units of the time step. 0 if absent. |
    | `requires` | What the program needs from the thermodynamic state. |
    | `provides` | Properties of the program. |
    | `effects` | The effects of the body: `{read = [...], write = [...]}`, the names of the resources. Unknown if absent. |

    `effects` is the summary that `dyn.step` takes for the program (D355,
    #673). The verifier checks it against the body: every effect of an op of
    the body is in it, on its resource, and no op of the body has unknown
    effects. A `func.call` has the effects that its function declares in
    `dyn.effects`, in the same form; a call of a function without them has
    unknown effects. The summary may name more than the body does.

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
    | `requires = ["pressure"]`, `provides = ["barostatting"]` | If the body has a `dyn.scale_positions` |
    | `provides = ["symplectic"]`, `["time_reversible"]` | Only if the body has neither |
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
    OptionalAttr<StrArrayAttr>:$provides,
    OptionalAttr<DictionaryAttr>:$effects
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
    A step has the effects that its program states in `effects`, which the
    verifier of the program checks against its body (D355,
    #673): a program that calls the host, or writes a state of the host, is
    not moved across another step or removed (docs/dynamics.md, Section
    2.2.3). A step of a program without `effects` reads and writes every
    resource.

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
