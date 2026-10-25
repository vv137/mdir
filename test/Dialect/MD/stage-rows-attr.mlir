// RUN: mdir-opt %s -split-input-file -verify-diagnostics

module {
  func.func @valid_tensor(%input: tensor<4x3xf32>) attributes {md.stage_rows = 4 : i64} {
    return
  }
}

// -----

module {
  func.func @valid_buffer(%input: memref<4x3xf32>) attributes {md.stage_rows = 4 : i64} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{expects 'md.stage_rows' as an i64 attribute on a function}}
  func.func @bad_type(%input: tensor<4x3xf32>) attributes {md.stage_rows = 4 : i32} {
    return
  }
}

// -----

// expected-error@+1 {{expects 'md.stage_rows' as an i64 attribute on a function}}
module attributes {md.stage_rows = 4 : i64} {
}

// -----

module {
  // expected-error@+1 {{'md.stage_rows' is 5, expected 4 rows from the first input}}
  func.func @wrong_rows(%input: tensor<4x3xf32>) attributes {md.stage_rows = 5 : i64} {
    return
  }
}
