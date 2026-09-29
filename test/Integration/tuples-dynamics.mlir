// Chains of particles with bonds, angles, and dihedrals integrated for 200
// steps of 0.002 with velocity Verlet, compiled and run, compared with the
// same integration in Inputs/tuples_reference.py. This file is generated
// from that script.
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

// The system is that of tuples.mlir, with the masses 1, 1.5, and 2 in
// turn and velocities from a linear congruential generator. Each check
// prints 1 if the value agrees with the reference to a relative tolerance
// of 1e-9, or 0 if it does not. The total energy deviates from its start by
// at most 4.0e-02 along the way, 1.5e-04 of its value.

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


dyn.program @velocity_verlet(%x: !vec, %v: !vec, %f: !vec, %m: !md.field<@atoms, f64>,
                             %cell: !md.cell, %dt: f64,
                             %bonds: !bonds, %kb: !of_bond, %r0: !of_bond,
                             %angles: !angles, %ka: !of_angle, %c0: !of_angle,
                             %dihedrals: !dihedrals, %kd: !of_dihedral,
                             %phi0: !of_dihedral) -> (!vec, !vec, !vec)
    attributes {provides = ["symplectic", "time_reversible"]} {
  %c    = arith.constant 0.5 : f64
  %half = arith.mulf %c, %dt : f64
  %v1 = dyn.kick %v, %f, %m, %half : !vec
  %x1 = dyn.drift %x, %v1, %dt : !vec
  %f1 = md.evaluate @bonded(%x1, %cell, %bonds, %kb, %r0, %angles, %ka, %c0,
                            %dihedrals, %kd, %phi0) request [forces]
      : (!vec, !md.cell, !bonds, !of_bond, !of_bond, !angles, !of_angle,
         !of_angle, !dihedrals, !of_dihedral, !of_dihedral) -> !vec
  %v2 = dyn.kick %v1, %f1, %m, %half : !vec
  dyn.return %x1, %v2, %f1 : !vec, !vec, !vec
}

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%value: f64, %reference: f64) {
  %tolerance = arith.constant 1.0e-9 : f64
  %difference = arith.subf %value, %reference : f64
  %error = math.absf %difference : f64
  %scale = math.absf %reference : f64
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

memref.global "private" constant @velocities : memref<48x3xf64> =
    dense<[[0.09617261368677849, -0.1798665155397935, -0.2479137587102337], [0.19108996273846263, 0.5136532271539586, -0.4768810048440678], [0.4767366970320129, 0.4910502364186363, -0.33590285337737036], [-0.16784355217694408, -0.13693295602893663, 0.1944315368568318], [0.4033776369308018, 0.31049257659146356, -0.5567120550872965], [-0.4910271026763237, -0.04622574613636567, 0.14229841719174552], [0.08354564443127149, -0.2514647817911787, -0.5718863424586339], [0.06269108516991967, -0.3241224586704953, 0.32550379915887284], [0.05031873438403838, -0.0328301829027219, 0.3859858145264702], [0.05678667979211443, 0.32880948207134175, -0.1184424347916825], [-0.07445564247771269, -0.08514775659164621, 0.23884694793054628], [-0.363836599384538, 0.35548247585797477, 0.15489737412685323], [0.13253547590122455, -0.37439082048109007, -0.2736633619966192], [0.41186142260105246, 0.13292671649509835, -0.1645882371554358], [-0.47561446517809397, 0.3987294347284155, -0.431683309816031], [-0.020963955541244812, -0.06998809796964957, -0.36278406621163917], [-0.3345488406355596, -0.3731377576219125, -0.3056176039131565], [0.3145445111569845, -0.24191723668223453, 0.008158743407370314], [0.10316077865556711, 0.39776668001690674, 0.06697120659777688], [-0.17022424364565975, -0.46630213593339753, -0.06328871750479771], [-0.2299582440043903, 0.49300322823950815, -0.08821047979613973], [0.14927970537812346, 0.11914550049954818, -0.11031568606591059], [0.03671797796980374, -0.07946964117905332, 0.15951266468295622], [-0.37870034562527305, -0.21898009625470471, 0.14664370848590302], [0.2706142306690001, -0.47753408222666216, -0.02892747472247316], [0.01748311625690096, -0.21512467690950465, 0.16993989425504374], [0.4172130661447429, 0.36919494394937324, -0.2596683525221629], [-0.2791566009271062, -0.4001650567839129, 0.037939210448207125], [0.4859442372237229, 0.46155243881771135, 0.31755201753953266], [-0.1915339425775326, -0.10398403645700051, 0.2483419310560243], [0.2661461538066053, -0.41101852196475697, -0.2501989011394067], [-0.31649876745521194, -0.44525845718777013, 0.24339359908157754], [0.49502318877623314, -0.09163606955876781, -0.03470292265733911], [0.17465477170319194, -0.348255312975703, -0.3056568130592091], [0.4241320366402053, -0.20935707490166855, -0.08732035613825752], [0.17568789249182576, 0.1919285338630693, -0.21632652634030414], [-0.3413352969671703, -0.39459691799452734, 0.07830201697328852], [-0.18264296731083757, -0.3542681382710321, 0.315425904493572], [-0.35034492179854876, 0.25224639710763264, 0.41987428397664595], [-0.27522759871660835, 0.4367580265065448, 0.3413640735929625], [-0.32177957739784485, 0.27709854722747373, -0.051852486536113754], [-0.010674378705314465, -0.16694009580856395, -0.011233154658435121], [-0.04194153227015502, 0.20650225554385948, -0.16518753252199125], [-0.048866269980660744, -0.42466720749831033, 0.1893140049909966], [-0.0273447896923042, 0.1685012332422452, 0.39939611640551853], [-0.37963132267921335, 0.409251599598469, -0.08717579578256442], [-0.26798214703901774, -0.18010991203805637, 0.005673534728379712], [0.43888833759249085, -0.14870197695886922, 0.17952684222513604]]>

memref.global "private" constant @masses : memref<48xf64> =
    dense<[1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0]>

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c20 = arith.constant 20 : index
  %steps = arith.constant 200 : index

  %xs = memref.get_global @positions : memref<48x3xf64>
  %xd = memref.cast %xs : memref<48x3xf64> to memref<?x3xf64>
  %n = memref.dim %xd, %c0 : memref<?x3xf64>
  %positions = memref.alloc(%n) : memref<?x3xf64>
  memref.copy %xd, %positions : memref<?x3xf64> to memref<?x3xf64>
  %x0 = mdrt.from_buffer %positions : memref<?x3xf64> to !vec

  %vs = memref.get_global @velocities : memref<48x3xf64>
  %vd = memref.cast %vs : memref<48x3xf64> to memref<?x3xf64>
  %velocities = memref.alloc(%n) : memref<?x3xf64>
  memref.copy %vd, %velocities : memref<?x3xf64> to memref<?x3xf64>
  %v0 = mdrt.from_buffer %velocities : memref<?x3xf64> to !vec

  %ms = memref.get_global @masses : memref<48xf64>
  %md = memref.cast %ms : memref<48xf64> to memref<?xf64>
  %m = mdrt.from_buffer %md : memref<?xf64> to !md.field<@atoms, f64>

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
  %dt = arith.constant 0.002 : f64

  %f0 = md.evaluate @bonded(%x0, %cell, %bonds, %bonds_k, %bonds_0,
                            %angles, %angles_k, %angles_0,
                            %dihedrals, %dihedrals_k, %dihedrals_0)
      request [forces]
      : (!vec, !md.cell, !bonds, !of_bond, !of_bond, !angles, !of_angle,
         !of_angle, !dihedrals, !of_dihedral, !of_dihedral) -> !vec

  %x, %v, %f = scf.for %s = %c0 to %steps step %c1
      iter_args(%xa = %x0, %va = %v0, %fa = %f0) -> (!vec, !vec, !vec) {
    %xb, %vb, %fb = dyn.step @velocity_verlet(
        %xa, %va, %fa, %m, %cell, %dt, %bonds, %bonds_k, %bonds_0,
        %angles, %angles_k, %angles_0, %dihedrals, %dihedrals_k, %dihedrals_0)
        : (!vec, !vec, !vec, !md.field<@atoms, f64>, !md.cell, f64, !bonds,
           !of_bond, !of_bond, !angles, !of_angle, !of_angle, !dihedrals,
           !of_dihedral, !of_dihedral) -> (!vec, !vec, !vec)
    scf.yield %xb, %vb, %fb : !vec, !vec, !vec
  }

  %u = md.evaluate @bonded(%x, %cell, %bonds, %bonds_k, %bonds_0,
                           %angles, %angles_k, %angles_0,
                           %dihedrals, %dihedrals_k, %dihedrals_0)
      request [energy]
      : (!vec, !md.cell, !bonds, !of_bond, !of_bond, !angles, !of_angle,
         !of_angle, !dihedrals, !of_dihedral, !of_dihedral) -> f64
  %k = md.sum_particles gather(%v, %m : !vec, !md.field<@atoms, f64>) {
  ^bb0(%v_i: vector<3xf64>, %m_i: f64):
    %half = arith.constant 0.5 : f64
    %sq   = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2   = vector.reduction <add>, %sq : vector<3xf64> into f64
    %mv2  = arith.mulf %m_i, %v2 : f64
    %ke   = arith.mulf %half, %mv2 : f64
    md.yield %ke : f64
  } : f64

  // The potential and the kinetic energy at the end.
  // CHECK:      1
  // CHECK-NEXT: 1
  %u_ref = arith.constant 139.09778002841364 : f64
  %k_ref = arith.constant 132.01263664539871 : f64
  call @check(%u, %u_ref) : (f64, f64) -> ()
  call @check(%k, %k_ref) : (f64, f64) -> ()

  // The position of particle 0 and the velocity of particle 20.
  // CHECK-COUNT-6: 1
  %xf = mdrt.to_buffer %x : !vec to memref<?x3xf64>
  %vf = mdrt.to_buffer %v : !vec to memref<?x3xf64>
  %x0_0 = memref.load %xf[%c0, %c0] : memref<?x3xf64>
  %x0_0_ref = arith.constant 3.366528755442555 : f64
  call @check(%x0_0, %x0_0_ref) : (f64, f64) -> ()
  %x0_1 = memref.load %xf[%c0, %c1] : memref<?x3xf64>
  %x0_1_ref = arith.constant 2.8774847569762634 : f64
  call @check(%x0_1, %x0_1_ref) : (f64, f64) -> ()
  %x0_2 = memref.load %xf[%c0, %c2] : memref<?x3xf64>
  %x0_2_ref = arith.constant 2.800295368315258 : f64
  call @check(%x0_2, %x0_2_ref) : (f64, f64) -> ()
  %v20_0 = memref.load %vf[%c20, %c0] : memref<?x3xf64>
  %v20_0_ref = arith.constant 1.7643927756421176 : f64
  call @check(%v20_0, %v20_0_ref) : (f64, f64) -> ()
  %v20_1 = memref.load %vf[%c20, %c1] : memref<?x3xf64>
  %v20_1_ref = arith.constant 1.295056220030035 : f64
  call @check(%v20_1, %v20_1_ref) : (f64, f64) -> ()
  %v20_2 = memref.load %vf[%c20, %c2] : memref<?x3xf64>
  %v20_2_ref = arith.constant -0.11946584904159044 : f64
  call @check(%v20_2, %v20_2_ref) : (f64, f64) -> ()
  // CHECK-NOT: 0
  return
}
