// Lennard-Jones between the particles of the chains of tuples.mlir, with
// the pairs one, two, and three bonds apart excluded from the neighborhood
// and the pairs three bonds apart added back with the factor 0.5,
// compiled and run, compared with Inputs/tuples_reference.py. This file is
// generated from that script by Inputs/generate_tuples_tests.py.
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

// eps = 0.5, sigma = 0.25, cutoff 1.5, with no truncation. Each
// check prints 1 if the value agrees with the reference to a relative
// tolerance of 1e-10, or 0 if it does not.

!vec      = !md.field<@atoms, 3 x f64>
!pairs    = !md.relation<@atoms, 2, unordered>
!excluded = !md.relation<@atoms, 2, unordered, @excluded>
!pairs14  = !md.relation<@atoms, 2, unordered, @pairs14>

md.particle_set @atoms
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @pairs14 on(@atoms) arity(2) orientation(unordered)

md.potential @nonbonded(%x: !vec, %cell: !md.cell, %e: !excluded,
                        %p: !pairs14, %eps: f64, %sigma: f64,
                        %scale: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5) exclude(%e : !excluded)
         : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u14 = md.sum_tuples %p, %x, %cell coordinates(distance(0, 1)) {
  ^bb0(%r: f64):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    %ks  = arith.mulf %scale, %k : f64
    md.yield %ks : f64
  } : !pairs14, !vec -> f64
  %s = arith.addf %u, %u14 : f64
  md.return %s : f64
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

memref.global "private" constant @excluded_members : memref<96x2xi32> =
    dense<[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5], [6, 7], [6, 8], [6, 9], [7, 8], [7, 9], [7, 10], [8, 9], [8, 10], [8, 11], [9, 10], [9, 11], [10, 11], [12, 13], [12, 14], [12, 15], [13, 14], [13, 15], [13, 16], [14, 15], [14, 16], [14, 17], [15, 16], [15, 17], [16, 17], [18, 19], [18, 20], [18, 21], [19, 20], [19, 21], [19, 22], [20, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23], [24, 25], [24, 26], [24, 27], [25, 26], [25, 27], [25, 28], [26, 27], [26, 28], [26, 29], [27, 28], [27, 29], [28, 29], [30, 31], [30, 32], [30, 33], [31, 32], [31, 33], [31, 34], [32, 33], [32, 34], [32, 35], [33, 34], [33, 35], [34, 35], [36, 37], [36, 38], [36, 39], [37, 38], [37, 39], [37, 40], [38, 39], [38, 40], [38, 41], [39, 40], [39, 41], [40, 41], [42, 43], [42, 44], [42, 45], [43, 44], [43, 45], [43, 46], [44, 45], [44, 46], [44, 47], [45, 46], [45, 47], [46, 47]]>

memref.global "private" constant @pairs14_members : memref<24x2xi32> =
    dense<[[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11], [12, 15], [13, 16], [14, 17], [18, 21], [19, 22], [20, 23], [24, 27], [25, 28], [26, 29], [30, 33], [31, 34], [32, 35], [36, 39], [37, 40], [38, 41], [42, 45], [43, 46], [44, 47]]>

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %xs = memref.get_global @positions : memref<48x3xf64>
  %xd = memref.cast %xs : memref<48x3xf64> to memref<?x3xf64>
  %n = memref.dim %xd, %c0 : memref<?x3xf64>
  %buffer = memref.alloc(%n) : memref<?x3xf64>
  memref.copy %xd, %buffer : memref<?x3xf64> to memref<?x3xf64>
  %x = mdrt.from_buffer %buffer : memref<?x3xf64> to !vec

  %es = memref.get_global @excluded_members : memref<96x2xi32>
  %ed = memref.cast %es : memref<96x2xi32> to memref<?x2xi32>
  %e = mdrt.from_buffer %ed : memref<?x2xi32> to !excluded
  %ps = memref.get_global @pairs14_members : memref<24x2xi32>
  %pd = memref.cast %ps : memref<24x2xi32> to memref<?x2xi32>
  %p = mdrt.from_buffer %pd : memref<?x2xi32> to !pairs14

  %edge = arith.constant 6.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %eps = arith.constant 0.5 : f64
  %sigma = arith.constant 0.25 : f64
  %scale = arith.constant 0.5 : f64

  %u, %f, %w = md.evaluate @nonbonded(%x, %cell, %e, %p, %eps, %sigma, %scale)
      request [energy, forces, virial]
      : (!vec, !md.cell, !excluded, !pairs14, f64, f64, f64)
        -> (f64, !vec, vector<9xf64>)
  %forces = mdrt.to_buffer %f : !vec to memref<?x3xf64>

  // Energy.
  // CHECK:      1
  %u_ref = arith.constant 1.7100542438599273 : f64
  call @check(%u, %u_ref, %u_ref) : (f64, f64, f64) -> ()

  // The force on particle 0, on the scale of its largest component.
  // CHECK-COUNT-3: 1
  %f0_scale = arith.constant 0.17976630013340833 : f64
  %f0x = memref.load %forces[%c0, %c0] : memref<?x3xf64>
  %f0y = memref.load %forces[%c0, %c1] : memref<?x3xf64>
  %f0z = memref.load %forces[%c0, %c2] : memref<?x3xf64>
  %f0x_ref = arith.constant -0.17976630013340833 : f64
  %f0y_ref = arith.constant 0.13728218669455455 : f64
  %f0z_ref = arith.constant 0.09928617196564614 : f64
  call @check(%f0x, %f0x_ref, %f0_scale) : (f64, f64, f64) -> ()
  call @check(%f0y, %f0y_ref, %f0_scale) : (f64, f64, f64) -> ()
  call @check(%f0z, %f0z_ref, %f0_scale) : (f64, f64, f64) -> ()

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
  %f2_ref = arith.constant 95382.6724527377 : f64
  call @check(%f2, %f2_ref, %f2_ref) : (f64, f64, f64) -> ()

  // The virial, every component on the scale of the largest.
  // CHECK-COUNT-9: 1
  %w_scale = arith.constant 36.29901886821435 : f64
  %w0 = vector.extract %w[0] : f64 from vector<9xf64>
  %w0_ref = arith.constant 8.321851197779813 : f64
  call @check(%w0, %w0_ref, %w_scale) : (f64, f64, f64) -> ()
  %w1 = vector.extract %w[1] : f64 from vector<9xf64>
  %w1_ref = arith.constant -19.27377172732078 : f64
  call @check(%w1, %w1_ref, %w_scale) : (f64, f64, f64) -> ()
  %w2 = vector.extract %w[2] : f64 from vector<9xf64>
  %w2_ref = arith.constant 5.1485067596015055 : f64
  call @check(%w2, %w2_ref, %w_scale) : (f64, f64, f64) -> ()
  %w3 = vector.extract %w[3] : f64 from vector<9xf64>
  %w3_ref = arith.constant -19.27377172732078 : f64
  call @check(%w3, %w3_ref, %w_scale) : (f64, f64, f64) -> ()
  %w4 = vector.extract %w[4] : f64 from vector<9xf64>
  %w4_ref = arith.constant 36.29901886821435 : f64
  call @check(%w4, %w4_ref, %w_scale) : (f64, f64, f64) -> ()
  %w5 = vector.extract %w[5] : f64 from vector<9xf64>
  %w5_ref = arith.constant -10.858535808896438 : f64
  call @check(%w5, %w5_ref, %w_scale) : (f64, f64, f64) -> ()
  %w6 = vector.extract %w[6] : f64 from vector<9xf64>
  %w6_ref = arith.constant 5.1485067596015055 : f64
  call @check(%w6, %w6_ref, %w_scale) : (f64, f64, f64) -> ()
  %w7 = vector.extract %w[7] : f64 from vector<9xf64>
  %w7_ref = arith.constant -10.858535808896438 : f64
  call @check(%w7, %w7_ref, %w_scale) : (f64, f64, f64) -> ()
  %w8 = vector.extract %w[8] : f64 from vector<9xf64>
  %w8_ref = arith.constant 2.2874949570744683 : f64
  call @check(%w8, %w8_ref, %w_scale) : (f64, f64, f64) -> ()
  // CHECK-NOT: 0
  return
}
