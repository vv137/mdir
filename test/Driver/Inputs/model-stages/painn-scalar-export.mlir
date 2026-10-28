// Frozen emission fixture from the repository's actual staged-model exporter.
// Regenerate with scripts/mlip/painn-stages.py:
//   weights --synthetic 1 0 weights.npz --seed 3
//   export weights.npz model.mlir --rows 64 --stablehlo-opt stablehlo-opt --workdir work
// One feature, no message layers: embedding and readout tensor/linalg functions,
// including the exported readout VJP. Normal tests need no JAX or model download.
// PaiNN in stages, written by scripts/mlip/painn-stages.py export
// F = 1, L = 0, cutoff 5.0 A, 64 rows a tile
md.stage @embed forward(@embed.f) : (i32) -> vector<4xf64>
md.stage @readout forward(@readout.f) vjp(@readout.b) : (vector<4xf64>) -> f64
  func.func @embed.f(%arg0: tensor<64xi32>) -> (tensor<64x4xf32> ) {
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %c99 = arith.constant 99 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant dense<100> : tensor<i32>
    %cst_1 = arith.constant dense<[[2.04091907], [-2.55566502], [0.418098837], [-0.567769587], [-0.452649295], [-0.215597168], [-2.01998615], [-0.231932372], [-0.865213096], [3.32299948], [0.225786611], [-0.352630794], [-0.281287432], [-0.668046355], [-1.05515051], [-0.390800983], [0.481945395], [-0.238553613], [0.957758724], [-0.19980213], [0.0242595654], [1.54582083], [0.545105517], [-0.505228758], [-0.182838976], [0.540525138], [1.93508804], [-0.269620329], [-0.243558675], [1.00231361], [-0.886459946], [-0.291720241], [0.882538974], [0.580350041], [9.151670e-02], [0.670104384], [-2.82816219], [1.02130687], [-0.959644734], [-1.66861987], [0.276445746], [0.700544893], [-0.444767445], [-1.07640588], [0.0261248332], [-0.0527473092], [1.40559816], [0.747407972], [0.193815649], [1.11163318], [-0.205523044], [-0.925899565], [0.584058285], [0.582538426], [-0.214828908], [-0.782808601], [0.229153901], [-2.49389434], [0.69012475], [0.491368264], [-1.63885713], [0.0613535084], [-0.964099645], [0.757221043], [-2.03416729], [-0.914494514], [7.095800e-01], [1.15640104], [-2.15800548], [-0.498039842], [0.328020096], [-0.609216153], [1.59064019], [-1.19122672], [0.354531944], [-1.04840553], [1.40596294], [-0.0216512289], [-0.372250557], [-1.71818495], [1.68182552], [0.752778589], [0.753563821], [1.13788128], [0.349226564], [-0.639246583], [-0.800241232], [-0.800199985], [1.37007236], [-1.46038115], [-0.596369505], [-0.321243912], [0.224619031], [0.575349391], [-1.24909699], [-1.73001349], [-0.00441423245], [1.2135638], [0.757058084], [0.215650782]]> : tensor<100x1xf32>
    %cst_2 = arith.constant dense<0> : tensor<i32>
    %0 = tensor.empty() : tensor<64xi32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%cst_2 : tensor<i32>) outs(%0 : tensor<64xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<64xi32>
    %2 = tensor.empty() : tensor<64xi1>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %1 : tensor<64xi32>, tensor<64xi32>) outs(%2 : tensor<64xi1>) {
    ^bb0(%in: i32, %in_3: i32, %out: i1):
      %18 = arith.cmpi slt, %in, %in_3 : i32
      linalg.yield %18 : i1
    } -> tensor<64xi1>
    %4 = tensor.empty() : tensor<64xi32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%cst_0 : tensor<i32>) outs(%4 : tensor<64xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<64xi32>
    %6 = tensor.empty() : tensor<64xi32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %5 : tensor<64xi32>, tensor<64xi32>) outs(%6 : tensor<64xi32>) {
    ^bb0(%in: i32, %in_3: i32, %out: i32):
      %18 = arith.addi %in, %in_3 : i32
      linalg.yield %18 : i32
    } -> tensor<64xi32>
    %8 = tensor.empty() : tensor<64xi32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3, %7, %arg0 : tensor<64xi1>, tensor<64xi32>, tensor<64xi32>) outs(%8 : tensor<64xi32>) {
    ^bb0(%in: i1, %in_3: i32, %in_4: i32, %out: i32):
      %18 = arith.select %in, %in_3, %in_4 : i32
      linalg.yield %18 : i32
    } -> tensor<64xi32>
    %10 = tensor.empty() : tensor<64x1xi32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<64xi32>) outs(%10 : tensor<64x1xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<64x1xi32>
    %12 = tensor.empty() : tensor<64x1xf32>
    %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%12 : tensor<64x1xf32>) {
    ^bb0(%out: f32):
      %18 = linalg.index 0 : index
      %extracted = tensor.extract %11[%18, %c0] : tensor<64x1xi32>
      %19 = arith.index_cast %extracted : i32 to index
      %20 = arith.maxsi %19, %c0 : index
      %21 = arith.minsi %20, %c99 : index
      %extracted_3 = tensor.extract %cst_1[%21, %c0] : tensor<100x1xf32>
      linalg.yield %extracted_3 : f32
    } -> tensor<64x1xf32>
    %14 = tensor.empty() : tensor<64x3xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%14 : tensor<64x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x3xf32>
    %16 = tensor.empty() : tensor<64x4xf32>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%16 : tensor<64x4xf32>) {
    ^bb0(%out: f32):
      %18 = linalg.index 0 : index
      %19 = linalg.index 1 : index
      %20 = arith.cmpi ult, %19, %c1 : index
      %21 = scf.if %20 -> (f32) {
        %extracted = tensor.extract %13[%18, %19] : tensor<64x1xf32>
        scf.yield %extracted : f32
      } else {
        %22 = arith.subi %19, %c1 : index
        %extracted = tensor.extract %15[%18, %22] : tensor<64x3xf32>
        scf.yield %extracted : f32
      }
      linalg.yield %21 : f32
    } -> tensor<64x4xf32>
    return %17 : tensor<64x4xf32>
  }

  func.func @readout.f(%arg0: tensor<64x4xf32>) -> (tensor<64xf32> ) {
    %cst = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<2625.49976> : tensor<f32>
    %cst_1 = arith.constant dense<-5.000000e-01> : tensor<f32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant dense<-0.317155629> : tensor<1x1xf32>
    %cst_4 = arith.constant dense<0.0293233693> : tensor<1xf32>
    %cst_5 = arith.constant dense<-0.243335083> : tensor<1x1xf32>
    %cst_6 = arith.constant dense<0.0817206576> : tensor<1xf32>
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [64, 1] [1, 1] : tensor<64x4xf32> to tensor<64x1xf32>
    %0 = tensor.empty() : tensor<64x1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %2 = linalg.matmul ins(%extracted_slice, %cst_3 : tensor<64x1xf32>, tensor<1x1xf32>) outs(%1 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %3 = tensor.empty() : tensor<1x1xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst_4 : tensor<1xf32>) outs(%3 : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1xf32>
    %5 = tensor.empty() : tensor<64x1xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<1x1xf32>) outs(%5 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x1xf32>
    %7 = tensor.empty() : tensor<64x1xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %6 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%7 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %40 = arith.addf %in, %in_7 : f32
      linalg.yield %40 : f32
    } -> tensor<64x1xf32>
    %9 = tensor.empty() : tensor<64x1xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<64x1xf32>) outs(%9 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %40 = arith.negf %in : f32
      linalg.yield %40 : f32
    } -> tensor<64x1xf32>
    %11 = tensor.empty() : tensor<64x1xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<64x1xf32>) outs(%11 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %40 = math.exp %in : f32
      linalg.yield %40 : f32
    } -> tensor<64x1xf32>
    %13 = tensor.empty() : tensor<64x1xf32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%13 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x1xf32>
    %15 = tensor.empty() : tensor<64x1xf32>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %12 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%15 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %40 = arith.addf %in, %in_7 : f32
      linalg.yield %40 : f32
    } -> tensor<64x1xf32>
    %17 = tensor.empty() : tensor<64x1xf32>
    %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%17 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x1xf32>
    %19 = tensor.empty() : tensor<64x1xf32>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %16 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%19 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %40 = arith.divf %in, %in_7 : f32
      linalg.yield %40 : f32
    } -> tensor<64x1xf32>
    %21 = tensor.empty() : tensor<64x1xf32>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %20 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%21 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %40 = arith.mulf %in, %in_7 : f32
      linalg.yield %40 : f32
    } -> tensor<64x1xf32>
    %23 = tensor.empty() : tensor<64x1xf32>
    %24 = linalg.fill ins(%cst_2 : f32) outs(%23 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %25 = linalg.matmul ins(%22, %cst_5 : tensor<64x1xf32>, tensor<1x1xf32>) outs(%24 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %26 = tensor.empty() : tensor<1x1xf32>
    %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst_6 : tensor<1xf32>) outs(%26 : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1xf32>
    %28 = tensor.empty() : tensor<64x1xf32>
    %29 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%27 : tensor<1x1xf32>) outs(%28 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x1xf32>
    %30 = tensor.empty() : tensor<64x1xf32>
    %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %29 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%30 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %40 = arith.addf %in, %in_7 : f32
      linalg.yield %40 : f32
    } -> tensor<64x1xf32>
    %collapsed = tensor.collapse_shape %31 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
    %32 = tensor.empty() : tensor<64xf32>
    %33 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%cst_1 : tensor<f32>) outs(%32 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64xf32>
    %34 = tensor.empty() : tensor<64xf32>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%collapsed, %33 : tensor<64xf32>, tensor<64xf32>) outs(%34 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %40 = arith.addf %in, %in_7 : f32
      linalg.yield %40 : f32
    } -> tensor<64xf32>
    %36 = tensor.empty() : tensor<64xf32>
    %37 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%cst_0 : tensor<f32>) outs(%36 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64xf32>
    %38 = tensor.empty() : tensor<64xf32>
    %39 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%35, %37 : tensor<64xf32>, tensor<64xf32>) outs(%38 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %40 = arith.mulf %in, %in_7 : f32
      linalg.yield %40 : f32
    } -> tensor<64xf32>
    return %39 : tensor<64xf32>
  }

  func.func @readout.b(%arg0: tensor<64x4xf32>, %arg1: tensor<64xf32>) -> (tensor<64x4xf32> ) {
    %cst = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<2625.49976> : tensor<f32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant dense<-0.317155629> : tensor<1x1xf32>
    %cst_3 = arith.constant dense<0.0293233693> : tensor<1xf32>
    %cst_4 = arith.constant dense<-0.243335083> : tensor<1x1xf32>
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [64, 1] [1, 1] : tensor<64x4xf32> to tensor<64x1xf32>
    %0 = tensor.empty() : tensor<64x1xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %2 = linalg.matmul ins(%extracted_slice, %cst_2 : tensor<64x1xf32>, tensor<1x1xf32>) outs(%1 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %3 = tensor.empty() : tensor<1x1xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<1xf32>) outs(%3 : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1xf32>
    %5 = tensor.empty() : tensor<64x1xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<1x1xf32>) outs(%5 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x1xf32>
    %7 = tensor.empty() : tensor<64x1xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %6 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%7 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %47 = arith.addf %in, %in_5 : f32
      linalg.yield %47 : f32
    } -> tensor<64x1xf32>
    %9 = tensor.empty() : tensor<64x1xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<64x1xf32>) outs(%9 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %47 = arith.negf %in : f32
      linalg.yield %47 : f32
    } -> tensor<64x1xf32>
    %11 = tensor.empty() : tensor<64x1xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<64x1xf32>) outs(%11 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %47 = math.exp %in : f32
      linalg.yield %47 : f32
    } -> tensor<64x1xf32>
    %13 = tensor.empty() : tensor<64x1xf32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%13 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x1xf32>
    %15 = tensor.empty() : tensor<64x1xf32>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %12 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%15 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %47 = arith.addf %in, %in_5 : f32
      linalg.yield %47 : f32
    } -> tensor<64x1xf32>
    %17 = tensor.empty() : tensor<64x1xf32>
    %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%17 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x1xf32>
    %19 = tensor.empty() : tensor<64x1xf32>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %16 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%19 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %47 = arith.divf %in, %in_5 : f32
      linalg.yield %47 : f32
    } -> tensor<64x1xf32>
    %21 = tensor.empty() : tensor<64x1xf32>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%21 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x1xf32>
    %23 = tensor.empty() : tensor<64x1xf32>
    %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %20 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%23 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %47 = arith.subf %in, %in_5 : f32
      linalg.yield %47 : f32
    } -> tensor<64x1xf32>
    %25 = tensor.empty() : tensor<64x1xf32>
    %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %24 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%25 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %47 = arith.mulf %in, %in_5 : f32
      linalg.yield %47 : f32
    } -> tensor<64x1xf32>
    %27 = tensor.empty() : tensor<64xf32>
    %28 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%cst_0 : tensor<f32>) outs(%27 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64xf32>
    %29 = tensor.empty() : tensor<64xf32>
    %30 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg1, %28 : tensor<64xf32>, tensor<64xf32>) outs(%29 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %47 = arith.mulf %in, %in_5 : f32
      linalg.yield %47 : f32
    } -> tensor<64xf32>
    %31 = tensor.empty() : tensor<64x1xf32>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%30 : tensor<64xf32>) outs(%31 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x1xf32>
    %33 = tensor.empty() : tensor<64x1xf32>
    %34 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%32, %cst_4 : tensor<64x1xf32>, tensor<1x1xf32>) outs(%34 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %47 = arith.mulf %in, %in_5 : f32
      %48 = arith.addf %out, %47 : f32
      linalg.yield %48 : f32
    } -> tensor<64x1xf32>
    %36 = tensor.empty() : tensor<64x1xf32>
    %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %35 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%36 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %47 = arith.mulf %in, %in_5 : f32
      linalg.yield %47 : f32
    } -> tensor<64x1xf32>
    %38 = tensor.empty() : tensor<64x1xf32>
    %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%35, %20 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%38 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %47 = arith.mulf %in, %in_5 : f32
      linalg.yield %47 : f32
    } -> tensor<64x1xf32>
    %40 = tensor.empty() : tensor<64x1xf32>
    %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%37, %26 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%40 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %47 = arith.mulf %in, %in_5 : f32
      linalg.yield %47 : f32
    } -> tensor<64x1xf32>
    %42 = tensor.empty() : tensor<64x1xf32>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%39, %41 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%42 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %47 = arith.addf %in, %in_5 : f32
      linalg.yield %47 : f32
    } -> tensor<64x1xf32>
    %44 = tensor.empty() : tensor<64x1xf32>
    %45 = linalg.fill ins(%cst_1 : f32) outs(%44 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %46 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%43, %cst_2 : tensor<64x1xf32>, tensor<1x1xf32>) outs(%45 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %47 = arith.mulf %in, %in_5 : f32
      %48 = arith.addf %out, %47 : f32
      linalg.yield %48 : f32
    } -> tensor<64x1xf32>
    %padded = tensor.pad %46 low[0, 0] high[0, 3] {
    ^bb0(%arg2: index, %arg3: index):
      tensor.yield %cst_1 : f32
    } : tensor<64x1xf32> to tensor<64x4xf32>
    return %padded : tensor<64x4xf32>
  }

md.potential @painn(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
                     %z: !md.field<@atoms, i32>) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(0.5) : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  %h0 = md.map_particles gather(%z : !md.field<@atoms, i32>) {
  ^bb0(%zi: i32):
    %e = md.apply_stage @embed(%zi) : (i32) -> vector<4xf64>
    md.yield %e : vector<4xf64>
  } : !md.field<@atoms, irreps<1x0e + 1x1o, ir_mul>, f64>
  %u = md.sum_particles gather(%h0 : !md.field<@atoms, irreps<1x0e + 1x1o, ir_mul>, f64>) {
  ^bb0(%hi: vector<4xf64>):
    %e = md.apply_stage @readout(%hi) : (vector<4xf64>) -> f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}
