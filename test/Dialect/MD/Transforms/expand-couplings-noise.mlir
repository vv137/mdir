// RUN: mdir-opt %s --dyn-expand-couplings | FileCheck %s

// dyn.noise expands to the text that the driver writes for the O step of
// Langevin dynamics (emitNoise in lib/Driver/Builder.cpp): the CHECK lines
// are the program @step of `mdir emit` for examples/argon/argon.toml under
// LANGEVIN with a friction of 1/ps, verbatim, but for the claims of the
// program, which a program with dyn.noise makes (docs/dynamics.md,
// Section 2.2.4).

md.particle_set @atoms
md.potential @energy(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %u = arith.constant 0.0 : f64
  md.return %u : f64
}

dyn.program @step(%arg0: !md.field<@atoms, 3 x f64>, %arg1: !md.field<@atoms, 3 x f64>, %arg2: !md.field<@atoms, 3 x f64>, %arg3: !md.field<@atoms, f64>, %arg4: !md.cell, %arg5: f64, %arg6: i64, %arg7: !md.field<@atoms, i32>) -> (!md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>) attributes {provides = ["thermostatting"], requires = ["temperature"]} {
  %cst = arith.constant 5.000000e-01 : f64
  %0 = arith.mulf %cst, %arg5 : f64
  %1 = dyn.kick %arg1, %arg2, %arg3, %0 : !md.field<@atoms, 3 x f64>
  %2 = dyn.drift %arg0, %1, %0 : !md.field<@atoms, 3 x f64>
  %3 = dyn.noise %1, %arg3, %arg6, %arg7 {decay = 0.99501247919268231 : f64, seed = 314159 : i64, spread = 0.12304325343822024 : f64} : !md.field<@atoms, 3 x f64>, !md.field<@atoms, i32>
  %4 = dyn.drift %2, %3, %0 : !md.field<@atoms, 3 x f64>
  %5 = md.evaluate @energy(%4, %arg4) request [forces] : (!md.field<@atoms, 3 x f64>, !md.cell) -> !md.field<@atoms, 3 x f64>
  %6 = dyn.kick %3, %5, %arg3, %0 : !md.field<@atoms, 3 x f64>
  dyn.return %4, %6, %5 : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>
}

// CHECK-LABEL: dyn.program @step(%arg0: !md.field<@atoms, 3 x f64>, %arg1: !md.field<@atoms, 3 x f64>, %arg2: !md.field<@atoms, 3 x f64>, %arg3: !md.field<@atoms, f64>, %arg4: !md.cell, %arg5: f64, %arg6: i64, %arg7: !md.field<@atoms, i32>) -> (!md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>) attributes {provides = ["thermostatting"], requires = ["temperature"]} {
// CHECK-NEXT: %cst = arith.constant 5.000000e-01 : f64
// CHECK-NEXT: %0 = arith.mulf %cst, %arg5 : f64
// CHECK-NEXT: %1 = dyn.kick %arg1, %arg2, %arg3, %0 : !md.field<@atoms, 3 x f64>
// CHECK-NEXT: %2 = dyn.drift %arg0, %1, %0 : !md.field<@atoms, 3 x f64>
// CHECK-NEXT: %3 = md.map_particles gather(%1, %arg3, %arg7 : !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64>, !md.field<@atoms, i32>) {
// CHECK-NEXT: ^bb0(%arg8: vector<3xf64>, %arg9: f64, %arg10: i32):
// CHECK-NEXT: %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT: %7 = arith.shrui %arg6, %c32_i64 : i64
// CHECK-NEXT: %8 = arith.trunci %arg6 : i64 to i32
// CHECK-NEXT: %9 = arith.trunci %7 : i64 to i32
// CHECK-NEXT: %c33554432_i32 = arith.constant 33554432 : i32
// CHECK-NEXT: %c-766435501_i32 = arith.constant -766435501 : i32
// CHECK-NEXT: %c-845247145_i32 = arith.constant -845247145 : i32
// CHECK-NEXT: %low, %high = arith.mului_extended %c-766435501_i32, %8 : i32
// CHECK-NEXT: %low_0, %high_1 = arith.mului_extended %c-845247145_i32, %arg10 : i32
// CHECK-NEXT: %c314159_i32 = arith.constant 314159 : i32
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %10 = arith.xori %high_1, %9 : i32
// CHECK-NEXT: %11 = arith.xori %10, %c314159_i32 : i32
// CHECK-NEXT: %12 = arith.xori %high, %c33554432_i32 : i32
// CHECK-NEXT: %13 = arith.xori %12, %c0_i32 : i32
// CHECK-NEXT: %low_2, %high_3 = arith.mului_extended %c-766435501_i32, %11 : i32
// CHECK-NEXT: %low_4, %high_5 = arith.mului_extended %c-845247145_i32, %13 : i32
// CHECK-NEXT: %c-1640217368_i32 = arith.constant -1640217368 : i32
// CHECK-NEXT: %c-1150833019_i32 = arith.constant -1150833019 : i32
// CHECK-NEXT: %14 = arith.xori %high_5, %low_0 : i32
// CHECK-NEXT: %15 = arith.xori %14, %c-1640217368_i32 : i32
// CHECK-NEXT: %16 = arith.xori %high_3, %low : i32
// CHECK-NEXT: %17 = arith.xori %16, %c-1150833019_i32 : i32
// CHECK-NEXT: %low_6, %high_7 = arith.mului_extended %c-766435501_i32, %15 : i32
// CHECK-NEXT: %low_8, %high_9 = arith.mului_extended %c-845247145_i32, %17 : i32
// CHECK-NEXT: %c1014218401_i32 = arith.constant 1014218401 : i32
// CHECK-NEXT: %c1993301258_i32 = arith.constant 1993301258 : i32
// CHECK-NEXT: %18 = arith.xori %high_9, %low_4 : i32
// CHECK-NEXT: %19 = arith.xori %18, %c1014218401_i32 : i32
// CHECK-NEXT: %20 = arith.xori %high_7, %low_2 : i32
// CHECK-NEXT: %21 = arith.xori %20, %c1993301258_i32 : i32
// CHECK-NEXT: %low_10, %high_11 = arith.mului_extended %c-766435501_i32, %19 : i32
// CHECK-NEXT: %low_12, %high_13 = arith.mului_extended %c-845247145_i32, %21 : i32
// CHECK-NEXT: %c-626313126_i32 = arith.constant -626313126 : i32
// CHECK-NEXT: %c842468239_i32 = arith.constant 842468239 : i32
// CHECK-NEXT: %22 = arith.xori %high_13, %low_8 : i32
// CHECK-NEXT: %23 = arith.xori %22, %c-626313126_i32 : i32
// CHECK-NEXT: %24 = arith.xori %high_11, %low_6 : i32
// CHECK-NEXT: %25 = arith.xori %24, %c842468239_i32 : i32
// CHECK-NEXT: %low_14, %high_15 = arith.mului_extended %c-766435501_i32, %23 : i32
// CHECK-NEXT: %low_16, %high_17 = arith.mului_extended %c-845247145_i32, %25 : i32
// CHECK-NEXT: %c2028122643_i32 = arith.constant 2028122643 : i32
// CHECK-NEXT: %c-308364780_i32 = arith.constant -308364780 : i32
// CHECK-NEXT: %26 = arith.xori %high_17, %low_12 : i32
// CHECK-NEXT: %27 = arith.xori %26, %c2028122643_i32 : i32
// CHECK-NEXT: %28 = arith.xori %high_15, %low_10 : i32
// CHECK-NEXT: %29 = arith.xori %28, %c-308364780_i32 : i32
// CHECK-NEXT: %low_18, %high_19 = arith.mului_extended %c-766435501_i32, %27 : i32
// CHECK-NEXT: %low_20, %high_21 = arith.mului_extended %c-845247145_i32, %29 : i32
// CHECK-NEXT: %c387591116_i32 = arith.constant 387591116 : i32
// CHECK-NEXT: %c-1459197799_i32 = arith.constant -1459197799 : i32
// CHECK-NEXT: %30 = arith.xori %high_21, %low_16 : i32
// CHECK-NEXT: %31 = arith.xori %30, %c387591116_i32 : i32
// CHECK-NEXT: %32 = arith.xori %high_19, %low_14 : i32
// CHECK-NEXT: %33 = arith.xori %32, %c-1459197799_i32 : i32
// CHECK-NEXT: %low_22, %high_23 = arith.mului_extended %c-766435501_i32, %31 : i32
// CHECK-NEXT: %low_24, %high_25 = arith.mului_extended %c-845247145_i32, %33 : i32
// CHECK-NEXT: %c-1252940411_i32 = arith.constant -1252940411 : i32
// CHECK-NEXT: %c1684936478_i32 = arith.constant 1684936478 : i32
// CHECK-NEXT: %34 = arith.xori %high_25, %low_20 : i32
// CHECK-NEXT: %35 = arith.xori %34, %c-1252940411_i32 : i32
// CHECK-NEXT: %36 = arith.xori %high_23, %low_18 : i32
// CHECK-NEXT: %37 = arith.xori %36, %c1684936478_i32 : i32
// CHECK-NEXT: %low_26, %high_27 = arith.mului_extended %c-766435501_i32, %35 : i32
// CHECK-NEXT: %low_28, %high_29 = arith.mului_extended %c-845247145_i32, %37 : i32
// CHECK-NEXT: %c1401495358_i32 = arith.constant 1401495358 : i32
// CHECK-NEXT: %c534103459_i32 = arith.constant 534103459 : i32
// CHECK-NEXT: %38 = arith.xori %high_29, %low_24 : i32
// CHECK-NEXT: %39 = arith.xori %38, %c1401495358_i32 : i32
// CHECK-NEXT: %40 = arith.xori %high_27, %low_22 : i32
// CHECK-NEXT: %41 = arith.xori %40, %c534103459_i32 : i32
// CHECK-NEXT: %low_30, %high_31 = arith.mului_extended %c-766435501_i32, %39 : i32
// CHECK-NEXT: %low_32, %high_33 = arith.mului_extended %c-845247145_i32, %41 : i32
// CHECK-NEXT: %c-239036169_i32 = arith.constant -239036169 : i32
// CHECK-NEXT: %c-616729560_i32 = arith.constant -616729560 : i32
// CHECK-NEXT: %42 = arith.xori %high_33, %low_28 : i32
// CHECK-NEXT: %43 = arith.xori %42, %c-239036169_i32 : i32
// CHECK-NEXT: %44 = arith.xori %high_31, %low_26 : i32
// CHECK-NEXT: %45 = arith.xori %44, %c-616729560_i32 : i32
// CHECK-NEXT: %low_34, %high_35 = arith.mului_extended %c-766435501_i32, %43 : i32
// CHECK-NEXT: %low_36, %high_37 = arith.mului_extended %c-845247145_i32, %45 : i32
// CHECK-NEXT: %c-1879567696_i32 = arith.constant -1879567696 : i32
// CHECK-NEXT: %c-1767562579_i32 = arith.constant -1767562579 : i32
// CHECK-NEXT: %46 = arith.xori %high_37, %low_32 : i32
// CHECK-NEXT: %47 = arith.xori %46, %c-1879567696_i32 : i32
// CHECK-NEXT: %48 = arith.xori %high_35, %low_30 : i32
// CHECK-NEXT: %49 = arith.xori %48, %c-1767562579_i32 : i32
// CHECK-NEXT: %cst_38 = arith.constant 5.000000e-01 : f64
// CHECK-NEXT: %cst_39 = arith.constant 2.3283064365386963E-10 : f64
// CHECK-NEXT: %cst_40 = arith.constant -2.000000e+00 : f64
// CHECK-NEXT: %cst_41 = arith.constant 6.2831853071795862 : f64
// CHECK-NEXT: %50 = arith.uitofp %47 : i32 to f64
// CHECK-NEXT: %51 = arith.addf %50, %cst_38 : f64
// CHECK-NEXT: %52 = arith.mulf %51, %cst_39 : f64
// CHECK-NEXT: %53 = arith.uitofp %low_36 : i32 to f64
// CHECK-NEXT: %54 = arith.addf %53, %cst_38 : f64
// CHECK-NEXT: %55 = arith.mulf %54, %cst_39 : f64
// CHECK-NEXT: %56 = arith.uitofp %49 : i32 to f64
// CHECK-NEXT: %57 = arith.addf %56, %cst_38 : f64
// CHECK-NEXT: %58 = arith.mulf %57, %cst_39 : f64
// CHECK-NEXT: %59 = arith.uitofp %low_34 : i32 to f64
// CHECK-NEXT: %60 = arith.addf %59, %cst_38 : f64
// CHECK-NEXT: %61 = arith.mulf %60, %cst_39 : f64
// CHECK-NEXT: %62 = math.log %52 : f64
// CHECK-NEXT: %63 = arith.mulf %cst_40, %62 : f64
// CHECK-NEXT: %64 = math.sqrt %63 : f64
// CHECK-NEXT: %65 = arith.mulf %cst_41, %55 : f64
// CHECK-NEXT: %66 = math.cos %65 : f64
// CHECK-NEXT: %67 = arith.mulf %64, %66 : f64
// CHECK-NEXT: %68 = math.sin %65 : f64
// CHECK-NEXT: %69 = arith.mulf %64, %68 : f64
// CHECK-NEXT: %70 = math.log %58 : f64
// CHECK-NEXT: %71 = arith.mulf %cst_40, %70 : f64
// CHECK-NEXT: %72 = math.sqrt %71 : f64
// CHECK-NEXT: %73 = arith.mulf %cst_41, %61 : f64
// CHECK-NEXT: %74 = math.cos %73 : f64
// CHECK-NEXT: %75 = arith.mulf %72, %74 : f64
// CHECK-NEXT: %76 = vector.from_elements %67, %69, %75 : vector<3xf64>
// CHECK-NEXT: %cst_42 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT: %77 = arith.cmpf ogt, %arg9, %cst_42 : f64
// CHECK-NEXT: %cst_43 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT: %78 = arith.select %77, %arg9, %cst_43 : f64
// CHECK-NEXT: %cst_44 = arith.constant 0.12304325343822024 : f64
// CHECK-NEXT: %79 = math.sqrt %78 : f64
// CHECK-NEXT: %80 = arith.divf %cst_44, %79 : f64
// CHECK-NEXT: %81 = vector.broadcast %80 : f64 to vector<3xf64>
// CHECK-NEXT: %82 = arith.mulf %81, %76 : vector<3xf64>
// CHECK-NEXT: %cst_45 = arith.constant dense<0.99501247919268231> : vector<3xf64>
// CHECK-NEXT: %83 = arith.mulf %cst_45, %arg8 : vector<3xf64>
// CHECK-NEXT: %84 = arith.addf %83, %82 : vector<3xf64>
// CHECK-NEXT: %85 = arith.select %77, %84, %arg8 : vector<3xf64>
// CHECK-NEXT: md.yield %85 : vector<3xf64>
// CHECK-NEXT: } : !md.field<@atoms, 3 x f64>
// CHECK-NEXT: %4 = dyn.drift %2, %3, %0 : !md.field<@atoms, 3 x f64>
// CHECK-NEXT: %5 = md.evaluate @energy(%4, %arg4) request [forces] : (!md.field<@atoms, 3 x f64>, !md.cell) -> !md.field<@atoms, 3 x f64>
// CHECK-NEXT: %6 = dyn.kick %3, %5, %arg3, %0 : !md.field<@atoms, 3 x f64>
// CHECK-NEXT: dyn.return %4, %6, %5 : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>
// CHECK-NEXT: }
