// Energy, forces, and virial of bonds, angles, and dihedrals, compiled and
// run on the CPU, compared with Inputs/tuples_reference.py, which evaluates the
// same terms from their definitions and checks its forces against finite
// differences. This file is generated from that script.
//
// RUN: mdir-opt %s %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// RUN: mdir-opt %s %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_openmp \
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%openmp \
// RUN: | FileCheck %s
// The system: 8 chains of 6 particles in a periodic cube with the edge
// 6.0. Bonds k (r - r0)^2 / 2, angles k (cos(theta) - c0)^2 / 2, and
// dihedrals k (1 + cos(2 phi - phi0)), with parameters for each tuple.
//
// Each check prints 1 if the value agrees with the reference to a relative
// tolerance of 1e-10, or 0 if it does not.

!vec       = !md.field<@atoms, 3 x f64>
!bonds     = !md.relation<@atoms, 2, unordered, @bonds>
!angles    = !md.relation<@atoms, 3, reversal, @angles>
!dihedrals = !md.relation<@atoms, 4, reversal, @dihedrals>
!of_bond     = !md.field<@bonds, f64>
!of_angle    = !md.field<@angles, f64>
!of_dihedral = !md.field<@dihedrals, f64>

md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)
md.tuple_set @angles on(@atoms) arity(3) orientation(reversal)
md.tuple_set @dihedrals on(@atoms) arity(4) orientation(reversal)

md.potential @bonded(%x: !vec, %cell: !md.cell,
                     %bonds: !bonds, %kb: !of_bond, %r0: !of_bond,
                     %angles: !angles, %ka: !of_angle, %c0: !of_angle,
                     %dihedrals: !dihedrals, %kd: !of_dihedral,
                     %phi0: !of_dihedral) -> f64 {
  %ub = md.sum_tuples %bonds, %x, %cell coordinates(distance(0, 1))
          tuple(%kb, %r0 : !of_bond, !of_bond) {
  ^bb0(%r: f64, %k_t: f64, %r0_t: f64):
    %half = arith.constant 0.5 : f64
    %dr = arith.subf %r, %r0_t : f64
    %dr2 = arith.mulf %dr, %dr : f64
    %kh = arith.mulf %half, %k_t : f64
    %e = arith.mulf %kh, %dr2 : f64
    md.yield %e : f64
  } : !bonds, !vec -> f64

  %ua = md.sum_tuples %angles, %x, %cell coordinates(cosine(0, 1, 2))
          tuple(%ka, %c0 : !of_angle, !of_angle) {
  ^bb0(%c: f64, %k_t: f64, %c0_t: f64):
    %half = arith.constant 0.5 : f64
    %dc = arith.subf %c, %c0_t : f64
    %dc2 = arith.mulf %dc, %dc : f64
    %kh = arith.mulf %half, %k_t : f64
    %e = arith.mulf %kh, %dc2 : f64
    md.yield %e : f64
  } : !angles, !vec -> f64

  %ud = md.sum_tuples %dihedrals, %x, %cell coordinates(dihedral(0, 1, 2, 3))
          tuple(%kd, %phi0 : !of_dihedral, !of_dihedral) {
  ^bb0(%phi: f64, %k_t: f64, %phi0_t: f64):
    %one = arith.constant 1.0 : f64
    %two = arith.constant 2.0 : f64
    %p2 = arith.mulf %two, %phi : f64
    %a = arith.subf %p2, %phi0_t : f64
    %cos = math.cos %a : f64
    %s = arith.addf %one, %cos : f64
    %e = arith.mulf %k_t, %s : f64
    md.yield %e : f64
  } : !dihedrals, !vec -> f64

  %u0 = arith.addf %ub, %ua : f64
  %u = arith.addf %u0, %ud : f64
  md.return %u : f64
}

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%value: f64, %reference: f64, %magnitude: f64) {
  %tolerance = arith.constant 1.0e-10 : f64
  %difference = arith.subf %value, %reference : f64
  %error = math.absf %difference : f64
  %scale = math.absf %magnitude : f64
  %bound = arith.mulf %tolerance, %scale : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

memref.global "private" constant @positions : memref<48x3xf64> =
    dense<[[3.493845538236201, 3.118912495672703, 2.795858549885452], [3.738208435332411, 3.050874986532393, 2.384265935472874], [4.081443344939115, 3.1757708068070936, 2.6231896735303932], [4.117906033375022, 3.3152886215055, 2.057354912757062], [3.7431258980010074, 3.4678228735117647, 1.8921499578060397], [3.8844376354233803, 3.813856945925347, 1.587627984748731], [5.094812532886863, 2.4577514017000794, 0.19619107246398926], [5.176596564985629, 2.833571535570306, -0.23873820321602196], [5.1230356305991736, 2.7021254841690983, 0.3102325973944398], [5.4813409096366605, 2.8620400989567085, -0.04343424333668372], [5.87212889794681, 2.6277188930479416, -0.42303420886135096], [5.71843251040431, 2.324853560249839, -0.09666126401013148], [4.963670535944402, 1.572002811357379, 1.515761316753924], [4.715869374567647, 1.315574267890978, 1.7940667894907296], [4.4938005064203175, 1.4622788319602134, 1.4111808056593695], [4.122638489475119, 1.141840813126086, 1.2339083208826607], [4.0944782622782006, 1.1615722441767335, 1.652075884121615], [3.603340274061772, 1.4824589909577797, 1.6711846694050496], [2.7962466292083263, 0.11128488834947348, 2.8318013604730368], [2.2816812924962564, -0.11961101300110302, 2.6663963637009047], [2.620739039220103, -0.2624646433668818, 2.939810859505557], [2.2911671150512114, -0.17441813613289092, 3.3175306058618146], [1.8892711919233087, -0.4577823853614267, 3.0099911932949084], [1.587143998042056, -0.7093644013617897, 3.10753695782024], [4.682085958309472, 5.394970837980509, 0.04195096995681524], [4.277930596852081, 5.314494268361878, -0.3824568149930071], [4.336841663870741, 5.7623359570186805, -0.153724498959889], [4.824641508200454, 5.718319530485689, -0.4775588532885076], [5.17912262831393, 5.916381802955997, -0.23010241453162864], [5.179446275153573, 6.136140821345318, -0.7676050834057244], [5.915213422849774, 5.54339184332639, 4.119357135146856], [6.145161059060344, 5.641394566435115, 4.585889205677606], [6.522158236700493, 5.6551771949456615, 4.323171481440684], [6.911027436222348, 6.043373525088466, 4.391803030798663], [7.308574958055349, 5.98474420254608, 4.326755761915015], [7.545501547095817, 6.431236681789484, 4.624720214155734], [3.844283069483936, 3.2709519658237696, 1.9492831947281957], [3.7084367531491944, 2.8867731614526777, 1.7240885911844537], [3.824775168599942, 3.128758978561751, 1.300343850000235], [4.1311984586257156, 2.940688782221031, 1.5586539375959645], [3.90259915111151, 3.2444817887879873, 1.2995261774227234], [3.5864125003927496, 2.822650951385228, 1.2082856949029308], [3.46928758174181, 3.7413053093478084, 3.0647820252925158], [3.051745365694134, 3.3599549585119135, 2.9903900381504562], [3.107829377514127, 3.135284452076345, 3.5116187207085794], [3.2912906823037056, 2.8282702134117326, 3.7390145044317835], [3.750109966004526, 2.4878848257490955, 3.79509877991828], [4.296547633909411, 2.5355819296630555, 3.76661721223492]]>

memref.global "private" constant @bonds_members : memref<40x2xi32> =
    dense<[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [42, 43], [43, 44], [44, 45], [45, 46], [46, 47]]>

memref.global "private" constant @bonds_k : memref<40xf64> =
    dense<[100.0, 110.0, 120.0, 130.0, 140.0, 100.0, 110.0, 120.0, 130.0, 140.0, 100.0, 110.0, 120.0, 130.0, 140.0, 100.0, 110.0, 120.0, 130.0, 140.0, 100.0, 110.0, 120.0, 130.0, 140.0, 100.0, 110.0, 120.0, 130.0, 140.0, 100.0, 110.0, 120.0, 130.0, 140.0, 100.0, 110.0, 120.0, 130.0, 140.0]>

memref.global "private" constant @bonds_0 : memref<40xf64> =
    dense<[0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.45, 0.46, 0.47000000000000003, 0.48, 0.49]>

memref.global "private" constant @angles_members : memref<32x3xi32> =
    dense<[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [12, 13, 14], [13, 14, 15], [14, 15, 16], [15, 16, 17], [18, 19, 20], [19, 20, 21], [20, 21, 22], [21, 22, 23], [24, 25, 26], [25, 26, 27], [26, 27, 28], [27, 28, 29], [30, 31, 32], [31, 32, 33], [32, 33, 34], [33, 34, 35], [36, 37, 38], [37, 38, 39], [38, 39, 40], [39, 40, 41], [42, 43, 44], [43, 44, 45], [44, 45, 46], [45, 46, 47]]>

memref.global "private" constant @angles_k : memref<32xf64> =
    dense<[20.0, 21.0, 22.0, 23.0, 20.0, 21.0, 22.0, 23.0, 20.0, 21.0, 22.0, 23.0, 20.0, 21.0, 22.0, 23.0, 20.0, 21.0, 22.0, 23.0, 20.0, 21.0, 22.0, 23.0, 20.0, 21.0, 22.0, 23.0, 20.0, 21.0, 22.0, 23.0]>

memref.global "private" constant @angles_0 : memref<32xf64> =
    dense<[-0.5, -0.4, -0.3, -0.19999999999999996, -0.5, -0.4, -0.3, -0.19999999999999996, -0.5, -0.4, -0.3, -0.19999999999999996, -0.5, -0.4, -0.3, -0.19999999999999996, -0.5, -0.4, -0.3, -0.19999999999999996, -0.5, -0.4, -0.3, -0.19999999999999996, -0.5, -0.4, -0.3, -0.19999999999999996, -0.5, -0.4, -0.3, -0.19999999999999996]>

memref.global "private" constant @dihedrals_members : memref<24x4xi32> =
    dense<[[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [6, 7, 8, 9], [7, 8, 9, 10], [8, 9, 10, 11], [12, 13, 14, 15], [13, 14, 15, 16], [14, 15, 16, 17], [18, 19, 20, 21], [19, 20, 21, 22], [20, 21, 22, 23], [24, 25, 26, 27], [25, 26, 27, 28], [26, 27, 28, 29], [30, 31, 32, 33], [31, 32, 33, 34], [32, 33, 34, 35], [36, 37, 38, 39], [37, 38, 39, 40], [38, 39, 40, 41], [42, 43, 44, 45], [43, 44, 45, 46], [44, 45, 46, 47]]>

memref.global "private" constant @dihedrals_k : memref<24xf64> =
    dense<[3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0]>

memref.global "private" constant @dihedrals_0 : memref<24xf64> =
    dense<[0.0, 0.3, 0.6, 0.0, 0.3, 0.6, 0.0, 0.3, 0.6, 0.0, 0.3, 0.6, 0.0, 0.3, 0.6, 0.0, 0.3, 0.6, 0.0, 0.3, 0.6, 0.0, 0.3, 0.6]>

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c20 = arith.constant 20 : index

  %xs = memref.get_global @positions : memref<48x3xf64>
  %xd = memref.cast %xs : memref<48x3xf64> to memref<?x3xf64>
  %n = memref.dim %xd, %c0 : memref<?x3xf64>
  %buffer = memref.alloc(%n) : memref<?x3xf64>
  memref.copy %xd, %buffer : memref<?x3xf64> to memref<?x3xf64>
  %x = mdrt.from_buffer %buffer : memref<?x3xf64> to !vec

  %bonds_ms = memref.get_global @bonds_members : memref<40x2xi32>
  %bonds_md = memref.cast %bonds_ms : memref<40x2xi32> to memref<?x2xi32>
  %bonds = mdrt.from_buffer %bonds_md : memref<?x2xi32> to !bonds
  %bonds_k_s = memref.get_global @bonds_k : memref<40xf64>
  %bonds_k_d = memref.cast %bonds_k_s : memref<40xf64> to memref<?xf64>
  %bonds_k = mdrt.from_buffer %bonds_k_d : memref<?xf64> to !of_bond
  %bonds_0_s = memref.get_global @bonds_0 : memref<40xf64>
  %bonds_0_d = memref.cast %bonds_0_s : memref<40xf64> to memref<?xf64>
  %bonds_0 = mdrt.from_buffer %bonds_0_d : memref<?xf64> to !of_bond
  %angles_ms = memref.get_global @angles_members : memref<32x3xi32>
  %angles_md = memref.cast %angles_ms : memref<32x3xi32> to memref<?x3xi32>
  %angles = mdrt.from_buffer %angles_md : memref<?x3xi32> to !angles
  %angles_k_s = memref.get_global @angles_k : memref<32xf64>
  %angles_k_d = memref.cast %angles_k_s : memref<32xf64> to memref<?xf64>
  %angles_k = mdrt.from_buffer %angles_k_d : memref<?xf64> to !of_angle
  %angles_0_s = memref.get_global @angles_0 : memref<32xf64>
  %angles_0_d = memref.cast %angles_0_s : memref<32xf64> to memref<?xf64>
  %angles_0 = mdrt.from_buffer %angles_0_d : memref<?xf64> to !of_angle
  %dihedrals_ms = memref.get_global @dihedrals_members : memref<24x4xi32>
  %dihedrals_md = memref.cast %dihedrals_ms : memref<24x4xi32> to memref<?x4xi32>
  %dihedrals = mdrt.from_buffer %dihedrals_md : memref<?x4xi32> to !dihedrals
  %dihedrals_k_s = memref.get_global @dihedrals_k : memref<24xf64>
  %dihedrals_k_d = memref.cast %dihedrals_k_s : memref<24xf64> to memref<?xf64>
  %dihedrals_k = mdrt.from_buffer %dihedrals_k_d : memref<?xf64> to !of_dihedral
  %dihedrals_0_s = memref.get_global @dihedrals_0 : memref<24xf64>
  %dihedrals_0_d = memref.cast %dihedrals_0_s : memref<24xf64> to memref<?xf64>
  %dihedrals_0 = mdrt.from_buffer %dihedrals_0_d : memref<?xf64> to !of_dihedral
  %edge = arith.constant 6.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge

  %u, %f, %w = md.evaluate @bonded(%x, %cell, %bonds, %bonds_k, %bonds_0,
                                   %angles, %angles_k, %angles_0,
                                   %dihedrals, %dihedrals_k, %dihedrals_0)
      request [energy, forces, virial]
      : (!vec, !md.cell, !bonds, !of_bond, !of_bond, !angles, !of_angle,
         !of_angle, !dihedrals, !of_dihedral, !of_dihedral)
        -> (f64, !vec, vector<9xf64>)
  %forces = mdrt.to_buffer %f : !vec to memref<?x3xf64>

  // Energy.
  // CHECK:      1
  %u_ref = arith.constant 262.0437430598166 : f64
  call @check(%u, %u_ref, %u_ref) : (f64, f64, f64) -> ()

  // The force on particle 0, on the scale of the largest component.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %f0_scale = arith.constant 21.33793718503979 : f64
  %f0x = memref.load %forces[%c0, %c0] : memref<?x3xf64>
  %f0y = memref.load %forces[%c0, %c1] : memref<?x3xf64>
  %f0z = memref.load %forces[%c0, %c2] : memref<?x3xf64>
  %f0x_ref = arith.constant -21.33793718503979 : f64
  %f0y_ref = arith.constant 3.1528838006239948 : f64
  %f0z_ref = arith.constant -17.122010752088666 : f64
  call @check(%f0x, %f0x_ref, %f0_scale) : (f64, f64, f64) -> ()
  call @check(%f0y, %f0y_ref, %f0_scale) : (f64, f64, f64) -> ()
  call @check(%f0z, %f0z_ref, %f0_scale) : (f64, f64, f64) -> ()

  // The y component of the force on particle 20.
  // CHECK-NEXT: 1
  %f20y = memref.load %forces[%c20, %c1] : memref<?x3xf64>
  %f20y_ref = arith.constant -54.42382649480984 : f64
  call @check(%f20y, %f20y_ref, %f20y_ref) : (f64, f64, f64) -> ()

  // The sum over all particles of the squared force.
  // CHECK-NEXT: 1
  %zero = arith.constant 0.0 : f64
  %f2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (f64) {
    %fx = memref.load %forces[%i, %c0] : memref<?x3xf64>
    %fy = memref.load %forces[%i, %c1] : memref<?x3xf64>
    %fz = memref.load %forces[%i, %c2] : memref<?x3xf64>
    %xx = arith.mulf %fx, %fx : f64
    %yy = arith.mulf %fy, %fy : f64
    %zz = arith.mulf %fz, %fz : f64
    %xy = arith.addf %xx, %yy : f64
    %sq = arith.addf %xy, %zz : f64
    %next = arith.addf %acc, %sq : f64
    scf.yield %next : f64
  }
  %f2_ref = arith.constant 89806.01220552914 : f64
  call @check(%f2, %f2_ref, %f2_ref) : (f64, f64, f64) -> ()

  // The virial W = sum over the tuples and their members m of d_m0 (x) F_m,
  // on the scale of its largest component: every component.
  // CHECK-COUNT-9: 1
  %w_scale = arith.constant 128.6422104014591 : f64
  %w0 = vector.extract %w[0] : f64 from vector<9xf64>
  %w0_ref = arith.constant -15.71076427596098 : f64
  call @check(%w0, %w0_ref, %w_scale) : (f64, f64, f64) -> ()
  %w1 = vector.extract %w[1] : f64 from vector<9xf64>
  %w1_ref = arith.constant -13.966232638633839 : f64
  call @check(%w1, %w1_ref, %w_scale) : (f64, f64, f64) -> ()
  %w2 = vector.extract %w[2] : f64 from vector<9xf64>
  %w2_ref = arith.constant -16.61032086418607 : f64
  call @check(%w2, %w2_ref, %w_scale) : (f64, f64, f64) -> ()
  %w3 = vector.extract %w[3] : f64 from vector<9xf64>
  %w3_ref = arith.constant -13.966232638633844 : f64
  call @check(%w3, %w3_ref, %w_scale) : (f64, f64, f64) -> ()
  %w4 = vector.extract %w[4] : f64 from vector<9xf64>
  %w4_ref = arith.constant 25.093706034860865 : f64
  call @check(%w4, %w4_ref, %w_scale) : (f64, f64, f64) -> ()
  %w5 = vector.extract %w[5] : f64 from vector<9xf64>
  %w5_ref = arith.constant -3.7720114069486392 : f64
  call @check(%w5, %w5_ref, %w_scale) : (f64, f64, f64) -> ()
  %w6 = vector.extract %w[6] : f64 from vector<9xf64>
  %w6_ref = arith.constant -16.61032086418605 : f64
  call @check(%w6, %w6_ref, %w_scale) : (f64, f64, f64) -> ()
  %w7 = vector.extract %w[7] : f64 from vector<9xf64>
  %w7_ref = arith.constant -3.7720114069486153 : f64
  call @check(%w7, %w7_ref, %w_scale) : (f64, f64, f64) -> ()
  %w8 = vector.extract %w[8] : f64 from vector<9xf64>
  %w8_ref = arith.constant -128.6422104014591 : f64
  call @check(%w8, %w8_ref, %w_scale) : (f64, f64, f64) -> ()
  // CHECK-NOT: 0
  return
}
