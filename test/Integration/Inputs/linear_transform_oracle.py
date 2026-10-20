"""Generate an FFT primitive test and check it against NumPy's direct DFT.

The input to the backward transform is the half-spectrum of another real
grid. Its zero and (for an even last axis) Nyquist planes therefore obey
the two-dimensional Hermitian symmetry required by a real transform.
"""

import argparse
import sys

import numpy as np


def direct_forward(values):
    """The DFT definition, with no FFT library or factorization."""
    n0, n1, n2 = values.shape
    points = np.indices(values.shape)
    result = np.empty((n0, n1, n2 // 2 + 1), dtype=np.complex128)
    for k0 in range(n0):
        for k1 in range(n1):
            for k2 in range(n2 // 2 + 1):
                phase = k0 * points[0] / n0 + k1 * points[1] / n1
                phase = phase + k2 * points[2] / n2
                result[k0, k1, k2] = np.sum(values * np.exp(-2j * np.pi * phase))
    return result


def direct_backward(half, length):
    """The unnormalized inverse DFT of a valid Hermitian half-spectrum."""
    n0, n1, h = half.shape
    full = np.empty((n0, n1, length), dtype=np.complex128)
    full[:, :, :h] = half
    for k2 in range(h, length):
        for k0 in range(n0):
            for k1 in range(n1):
                full[k0, k1, k2] = np.conj(
                    half[(-k0) % n0, (-k1) % n1, length - k2]
                )
    indices = np.indices(full.shape)
    result = np.empty(full.shape, dtype=np.float64)
    for r0 in range(n0):
        for r1 in range(n1):
            for r2 in range(length):
                phase = r0 * indices[0] / n0 + r1 * indices[1] / n1
                phase = phase + r2 * indices[2] / length
                result[r0, r1, r2] = np.sum(
                    full * np.exp(2j * np.pi * phase)
                ).real
    return result


def data(length, precision):
    shape = (3, 2, length)
    dtype = np.float32 if precision == "mixed" else np.float64
    n = np.prod(shape)
    places = np.arange(n, dtype=np.float64)
    x = (np.sin(0.23 * places + 0.4) + 0.17 * np.cos(0.61 * places)).astype(dtype)
    y = (np.cos(0.31 * places - 0.2) - 0.13 * np.sin(0.47 * places)).astype(dtype)
    fy = direct_forward(y.reshape(shape)).astype(
        np.complex64 if precision == "mixed" else np.complex128
    )
    packed = np.stack((fy.real, fy.imag), axis=-1).reshape(-1).astype(dtype)
    return shape, x, y, packed


def emit(length, precision, backend):
    shape, x, _, packed = data(length, precision)
    n = x.size
    h = packed.size
    ty = "f32" if precision == "mixed" else "f64"
    host = f"memref<?x{ty}>"
    device = f"memref<?x{ty}, 1>"
    lines = ["func.func private @mdrtPrintFFTValue(f64)", "func.func @main() {",
             "  %c0 = arith.constant 0 : index",
             "  %c1 = arith.constant 1 : index",
             f"  %n = arith.constant {n} : index",
             f"  %h = arith.constant {h} : index",
             f"  %x = memref.alloc(%n) : {host}",
             f"  %s = memref.alloc(%h) : {host}",
             f"  %f = memref.alloc(%h) : {host}",
             f"  %b = memref.alloc(%n) : {host}"]
    for prefix, values in (("x", x), ("s", packed)):
        for i, value in enumerate(values):
            number = f"{float(value):.17g}"
            if "." not in number and "e" not in number:
                number += ".0"
            lines += [f"  %i{prefix}{i} = arith.constant {i} : index",
                      f"  %v{prefix}{i} = arith.constant {number} : {ty}",
                      f"  memref.store %v{prefix}{i}, %{prefix}[%i{prefix}{i}] : {host}"]
    if backend == "gpu":
        for name, count in (("dx", "n"), ("ds", "h"), ("df", "h"), ("db", "n")):
            lines.append(f"  %{name} = gpu.alloc (%{count}) : {device}")
        lines += ["  %t0 = gpu.wait async",
                  f"  %t1 = gpu.memcpy async [%t0] %dx, %x : {device}, {host}",
                  f"  %t2 = gpu.memcpy async [%t1] %ds, %s : {device}, {host}",
                  "  gpu.wait [%t2]"]
        source, spectrum, backward, result = "%dx", "%df", "%ds", "%db"
        buffer_type = device
    else:
        source, spectrum, backward, result = "%x", "%f", "%s", "%b"
        buffer_type = host
    grid = f"grid([{shape[0]}, {shape[1]}, {shape[2]}]) kind(\"fft3d\")"
    lines += [f"  md_exec.linear_transform {source} outs({spectrum} : {buffer_type})",
              f"    {grid} direction(\"forward\") adjoint(\"backward\")",
              f"    pairing(\"real_hermitian_half\") : {buffer_type}",
              f"  md_exec.linear_transform {backward} outs({result} : {buffer_type})",
              f"    {grid} direction(\"backward\") adjoint(\"forward\")",
              f"    pairing(\"real_hermitian_half\") : {buffer_type}"]
    if backend == "gpu":
        lines += ["  %t3 = gpu.wait async",
                  f"  %t4 = gpu.memcpy async [%t3] %f, %df : {host}, {device}",
                  f"  %t5 = gpu.memcpy async [%t4] %b, %db : {host}, {device}",
                  "  gpu.wait [%t5]"]
    for name, count in (("f", "h"), ("b", "n")):
        lines.append(f"  scf.for %i = %c0 to %{count} step %c1 {{")
        lines.append(f"    %v = memref.load %{name}[%i] : {host}")
        if ty == "f32":
            lines.append("    %w = arith.extf %v : f32 to f64")
            value = "%w"
        else:
            value = "%v"
        lines += [f"    func.call @mdrtPrintFFTValue({value}) : (f64) -> ()",
                  "  }"]
    lines += ["  return", "}"]
    print("\n".join(lines))


def check(length, precision):
    shape, x, y, packed = data(length, precision)
    words = sys.stdin.read().split()
    expected_count = packed.size + x.size
    if len(words) != expected_count:
        raise AssertionError(f"expected {expected_count} output values, got {len(words)}")
    actual = np.asarray([float(word) for word in words])
    forward = actual[:packed.size]
    backward = actual[packed.size:]
    fx = direct_forward(x.reshape(shape))
    reference_forward = np.stack((fx.real, fx.imag), axis=-1).reshape(-1)
    complex_y = packed.reshape(*fx.shape, 2)
    complex_y = complex_y[..., 0] + 1j * complex_y[..., 1]
    reference_backward = direct_backward(complex_y, length).reshape(-1)
    scale = max(1.0, float(np.max(np.abs(reference_forward))),
                float(np.max(np.abs(reference_backward))))
    tolerance = (1e-6 if precision == "mixed" else 2e-11) * scale
    forward_error = float(np.max(np.abs(forward - reference_forward)))
    backward_error = float(np.max(np.abs(backward - reference_backward)))
    fy = packed.reshape(*fx.shape, 2)
    actual_fx = forward.reshape(*fx.shape, 2)
    weights = np.ones(fx.shape[-1])
    weights[1:] = 2.0
    if length % 2 == 0:
        weights[-1] = 1.0
    spectral_dot = float(np.sum(weights * np.sum(actual_fx * fy, axis=-1)))
    real_dot = float(np.dot(x.astype(np.float64), backward))
    adjoint_error = abs(spectral_dot - real_dot)
    adjoint_tolerance = (1e-6 if precision == "mixed" else 2e-11) * max(
        1.0, abs(spectral_dot), abs(real_dot))
    if (forward_error > tolerance or backward_error > tolerance or
            adjoint_error > adjoint_tolerance):
        raise AssertionError(
            f"forward {forward_error}, backward {backward_error} "
            f"(tolerance {tolerance}); adjoint {adjoint_error} "
            f"(tolerance {adjoint_tolerance})"
        )
    print(f"forward_reference_max={np.max(np.abs(reference_forward)):.8g} "
          f"forward_error={forward_error:.8g} "
          f"backward_reference_max={np.max(np.abs(reference_backward)):.8g} "
          f"backward_error={backward_error:.8g} "
          f"value_tolerance={tolerance:.8g} "
          f"adjoint_reference={spectral_dot:.8g} "
          f"adjoint_error={adjoint_error:.8g} "
          f"adjoint_tolerance={adjoint_tolerance:.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("emit", "check"))
    parser.add_argument("--length", type=int, choices=(5, 6), required=True)
    parser.add_argument("--precision", choices=("mixed", "double"), required=True)
    parser.add_argument("--backend", choices=("host", "gpu"), default="host")
    args = parser.parse_args()
    if args.action == "emit":
        emit(args.length, args.precision, args.backend)
    else:
        check(args.length, args.precision)
