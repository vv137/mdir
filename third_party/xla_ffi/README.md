# xla_ffi

The headers of the XLA FFI (`xla/ffi/api/c_api.h`, `api.h`, `ffi.h`) by
the OpenXLA Authors, as the wheel of jaxlib 0.11.2 ships them in
`jaxlib/include/xla/ffi/api` (the directory of `jax.ffi.include_dir()`),
unchanged. Their FFI API version is 0.4 (`XLA_FFI_API_MAJOR`,
`XLA_FFI_API_MINOR` in `c_api.h`). They are under the Apache License 2.0
in `LICENSE`.

| File | SHA-256 |
|---|---|
| `xla/ffi/api/api.h` | `1426ea8206a1b9f1824f460572ea731665eb7cb9def1f0fc166b13df4b955ad0` |
| `xla/ffi/api/c_api.h` | `9847993912aa1aadb2dac4af16eeb9d8bae3f470905b519977d1d09b3f37eedf` |
| `xla/ffi/api/ffi.h` | `0878473a1ee35fdd9ff15424d46a44602fb53019f0d0a9f5117abdb386c35129` |

MDIR uses them for the handlers of the JAX adapter of the frame evaluator
(`python/XlaFfi.cpp`, `mdir.jax`, docs/python-frames.md), so that MDIR
builds without JAX. The handlers check the version of the API that XLA
gives them against these headers when they are registered and at each
instantiation (D[frame-evaluator-jax]).
