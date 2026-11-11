module {
  func.func private @columns3(vector<3xf64>, index, index) -> (index, index)
  func.func private @columns6(vector<6xf64>, index, index) -> (index, index)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func @show(%x: index, %y: index) {
    %a = arith.index_cast %x : index to i64
    %b = arith.index_cast %y : index to i64
    func.call @printI64(%a) : (i64) -> ()
    func.call @printNewline() : () -> ()
    func.call @printI64(%b) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return
  }
  func.func @main() {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : index
    %n = arith.constant 1024 : index
    %large = arith.constant 8192 : index
    %bins = arith.constant 512 : index
    %small = arith.constant 12 : index
    %a = arith.constant dense<[4.0, 4.0, 4.0]> : vector<3xf64>
    %b = arith.constant dense<[8.0, 8.0, 8.0]> : vector<3xf64>
    %tiny = arith.constant dense<[1.0, 1.0, 1.0]> : vector<3xf64>
    %flat = arith.constant dense<[1.0e12, 1.0e-6, 1.0e-6]> : vector<3xf64>
    %tilt = arith.constant dense<[4.0, 4.0, 4.0, 3.0, 2.0, 1.0]> : vector<6xf64>
    %x0, %y0 = func.call @columns3(%a, %n, %bins) : (vector<3xf64>, index, index) -> (index, index)
    func.call @show(%x0, %y0) : (index, index) -> ()
    %x1, %y1 = func.call @columns3(%a, %large, %bins) : (vector<3xf64>, index, index) -> (index, index)
    func.call @show(%x1, %y1) : (index, index) -> ()
    %x2, %y2 = func.call @columns3(%b, %large, %bins) : (vector<3xf64>, index, index) -> (index, index)
    func.call @show(%x2, %y2) : (index, index) -> ()
    %x3, %y3 = func.call @columns3(%b, %large, %small) : (vector<3xf64>, index, index) -> (index, index)
    func.call @show(%x3, %y3) : (index, index) -> ()
    %x4, %y4 = func.call @columns3(%tiny, %zero, %small) : (vector<3xf64>, index, index) -> (index, index)
    func.call @show(%x4, %y4) : (index, index) -> ()
    %x5, %y5 = func.call @columns3(%flat, %one, %small) : (vector<3xf64>, index, index) -> (index, index)
    func.call @show(%x5, %y5) : (index, index) -> ()
    %x6, %y6 = func.call @columns6(%tilt, %n, %bins) : (vector<6xf64>, index, index) -> (index, index)
    func.call @show(%x6, %y6) : (index, index) -> ()
    return
  }
}
