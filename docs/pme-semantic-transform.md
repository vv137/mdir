# Semantic linear transform for reciprocal sums

D319 (#502, a slice of #452) adds a pure `md` operation
for a linear transform whose transpose is declared in the IR. The first
kind is the three-dimensional real FFT. D306 supplies its separate
execution-level operation over explicit buffers. The semantic operation
has SSA tensor values so that a future reciprocal sum can compose and
differentiate its stages before choosing storage or a device. The scalar
type of each value selects mixed (`f32`) or double (`f64`) execution.
The particle-mesh composition follows [[Essmann1995]](references.md#essmann1995);
the transform contract itself follows the discrete Fourier definition.

## Values and mathematical contract

For positive grid extents $K_1,K_2,K_3$, `forward` maps a real grid of
shape $(K_1,K_2,K_3)$ to a complex half-spectrum of shape
$(K_1,K_2,\lfloor K_3/2\rfloor+1)$. Matching `f32`/`complex<f32>` and
`f64`/`complex<f64>` values are supported. With indices starting at zero,
$\hat x_{\mathbf k}=\sum_{\mathbf j}x_{\mathbf j}
e^{-2\pi i\sum_a j_a k_a/K_a}$; `backward` has the positive exponent and
no factor of $1/(K_1K_2K_3)$. Thus `backward(forward(x))` is
$(K_1K_2K_3)x$.

The half-spectrum is the stored half of the DFT of a real grid. Its
$k_3=0$ plane and, for even $K_3$, its Nyquist plane have Hermitian
symmetry in the first two axes. A user of `backward` promises this
condition; the verifier checks shape and type but cannot prove a tensor's
contents. An arbitrary complex tensor of the right shape is outside the
adjoint contract.

For valid half-spectra $y,z$, the real Hermitian pairing is
$\langle y,z\rangle_{\mathrm H}=\mathrm{Re}\sum_{\mathbf k\text{ stored}}
w_{k_3}\overline{y_{\mathbf k}}z_{\mathbf k}$, with $w_{k_3}=1$ at
zero and (for even $K_3$) Nyquist, and $w_{k_3}=2$ elsewhere. Parseval's
identity then gives $\langle\mathcal F x,y\rangle_{\mathrm H}
=\langle x,\mathcal F^\dagger y\rangle$ under the ordinary real-grid
dot product. This proves the declared transpose: the backward transform
for `forward`, and vice versa. The operation itself does no normalization;
a consumer wanting an inverse divides separately by $K_1K_2K_3$.

## IR contract

```mlir
%s = md.linear_transform %grid kind("fft3d") direction("forward")
  adjoint("backward") pairing("real_hermitian_half") :
  tensor<4x4x6xf64> -> tensor<4x4x4xcomplex<f64>>
```

The verifier requires three static extents, each from 2 through the
32-bit limit, and checks that their products and the packed execution
size fit in a signed 64-bit index. It checks the real and complex scalar
types, the first two matching extents, the half-spectrum's last extent,
`kind("fft3d")`, the two direction strings, the opposite declared
adjoint, and `pairing("real_hermitian_half")`. `AdjointOpInterface`
constructs the opposite `md.linear_transform` for a single result
cotangent; `--md-check-transposes` requires the declaration in a
potential. Invalid declarations receive an error rather than a zero
result.

This operation does not add a control-file or Python key and does not
change a trajectory. `md.reciprocal` still provides the potential,
forces, and virial. D[fft-value-buffer] (#526) lowers a semantic value to
the explicit buffers of D306: real grids use row-major scalar storage and
half-spectra use row-major interleaved real and imaginary scalars. Each
backward input is copied into an owned execution buffer because execution
may overwrite it; the semantic tensor remains immutable. Bufferization
and ownership-based deallocation precede optional device placement, which
copies only this transform's buffers and waits on those transfers. #452
still needs semantic spread and gather with their transposes, the
influence map, and differentiation of their composition. Distributed grid
transfer can then become a dependency between these primitive stages.
