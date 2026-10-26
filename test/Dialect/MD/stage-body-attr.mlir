// RUN: mdir-opt %s -split-input-file -verify-diagnostics

module {
  func.func @lowered_body(%input: memref<4xf32>) attributes {md.stage_body} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{expects 'md.stage_body' as a unit attribute on a function}}
  func.func @wrong_type() attributes {md.stage_body = true} {
    return
  }
}

// -----

// expected-error@+1 {{expects 'md.stage_body' as a unit attribute on a function}}
module attributes {md.stage_body} {
}

// -----

module {
  // expected-error@+1 {{expects 'md.stage_body' on a function with a body}}
  func.func private @external() attributes {md.stage_body}
}
