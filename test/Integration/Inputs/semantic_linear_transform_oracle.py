"""Exercise the semantic FFT bridge against an independent direct DFT."""

import argparse
import sys

import numpy as np

from linear_transform_oracle import check as check_execution, data


def emit(length, precision):
    shape, x, _, packed = data(length, precision)
    scalar = "f32" if precision == "mixed" else "f64"
    real_type = f"tensor<{shape[0]}x{shape[1]}x{shape[2]}x{scalar}>"
    half_type = f"tensor<{shape[0]}x{shape[1]}x{shape[2] // 2 + 1}xcomplex<{scalar}>>"
    lines = ["func.func private @mdrtPrintFFTValue(f64)", "func.func @main() {"]

    def constant(value, name):
        number = f"{float(value):.17g}"
        if "." not in number and "e" not in number:
            number += ".0"
        lines.append(f"  %{name} = arith.constant {number} : {scalar}")
        return f"%{name}"

    real = [constant(value, f"x{i}") for i, value in enumerate(x)]
    half = []
    for i, (re, im) in enumerate(packed.reshape(-1, 2)):
        r = constant(re, f"re{i}")
        j = constant(im, f"im{i}")
        lines.append(f"  %c{i} = complex.create {r}, {j} : complex<{scalar}>")
        half.append(f"%c{i}")
    lines.append(f"  %x = tensor.from_elements {', '.join(real)} : {real_type}")
    lines.append(f"  %s = tensor.from_elements {', '.join(half)} : {half_type}")
    lines += [f'  %f = md.linear_transform %x kind("fft3d") direction("forward")',
              f'    adjoint("backward") pairing("real_hermitian_half") : {real_type} -> {half_type}',
              f'  %b = md.linear_transform %s kind("fft3d") direction("backward")',
              f'    adjoint("forward") pairing("real_hermitian_half") : {half_type} -> {real_type}',
              f'  %again = md.linear_transform %s kind("fft3d") direction("backward")',
              f'    adjoint("forward") pairing("real_hermitian_half") : {half_type} -> {real_type}',
              f'  %roundtrip = md.linear_transform %f kind("fft3d") direction("backward")',
              f'    adjoint("forward") pairing("real_hermitian_half") : {half_type} -> {real_type}']

    def print_scalar(value, suffix):
        if scalar == "f32":
            lines.append(f"  %wide{suffix} = arith.extf {value} : f32 to f64")
            value = f"%wide{suffix}"
        lines.append(f"  func.call @mdrtPrintFFTValue({value}) : (f64) -> ()")

    for prefix, values, ty in (("f", packed.reshape(-1, 2), half_type),
                               ("b", x, real_type),
                               ("again", x, real_type),
                               ("roundtrip", x, real_type),
                               ("s", packed.reshape(-1, 2), half_type)):
        dims = shape if ty == real_type else (shape[0], shape[1], shape[2] // 2 + 1)
        for i in range(len(values)):
            row = i // (dims[1] * dims[2])
            col = (i // dims[2]) % dims[1]
            z = i % dims[2]
            indices = []
            for axis, number in enumerate((row, col, z)):
                name = f"i{prefix}{i}{axis}"
                lines.append(f"  %{name} = arith.constant {number} : index")
                indices.append(f"%{name}")
            suffix = f"{prefix}{i}"
            lines.append(f"  %v{suffix} = tensor.extract %{prefix}[{', '.join(indices)}] : {ty}")
            if ty == half_type:
                lines.append(f"  %r{suffix} = complex.re %v{suffix} : complex<{scalar}>")
                lines.append(f"  %j{suffix} = complex.im %v{suffix} : complex<{scalar}>")
                print_scalar(f"%r{suffix}", f"r{suffix}")
                print_scalar(f"%j{suffix}", f"j{suffix}")
            else:
                print_scalar(f"%v{suffix}", suffix)
    lines += ["  return", "}"]
    print("\n".join(lines))


def check(length, precision):
    shape, x, _, packed = data(length, precision)
    words = sys.stdin.read().split()
    count = packed.size + x.size
    if len(words) != count + 2 * x.size + packed.size:
        raise AssertionError(f"expected {count + 2 * x.size + packed.size} values, got {len(words)}")
    check_execution_input = " ".join(words[:count])
    # Reuse the direct-DFT and weighted-adjoint oracle for the first two results.
    import io
    old_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO(check_execution_input)
        check_execution(length, precision)
    finally:
        sys.stdin = old_stdin
    original = np.asarray([float(word) for word in words[count:]])
    tolerance = (1e-6 if precision == "mixed" else 2e-11) * max(1.0, float(np.max(np.abs(x))))
    repeat_error = float(np.max(np.abs(original[:x.size] - np.asarray([float(w) for w in words[packed.size:count]]))))
    roundtrip = original[x.size:2 * x.size]
    roundtrip_reference = x * np.prod(shape)
    roundtrip_tolerance = (1e-6 if precision == "mixed" else 2e-11) * max(
        1.0, float(np.max(np.abs(roundtrip_reference))))
    roundtrip_error = float(np.max(np.abs(roundtrip - roundtrip_reference)))
    input_error = float(np.max(np.abs(original[2 * x.size:] - packed)))
    if (repeat_error > tolerance or input_error > tolerance or
            roundtrip_error > roundtrip_tolerance):
        raise AssertionError(
            f"repeat {repeat_error}, immutable input {input_error}, "
            f"roundtrip {roundtrip_error}, tolerance {roundtrip_tolerance}")
    print(f"repeat_error={repeat_error:.8g} input_error={input_error:.8g} "
          f"roundtrip_error={roundtrip_error:.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("emit", "check"))
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--precision", choices=("mixed", "double"), required=True)
    args = parser.parse_args()
    (emit if args.action == "emit" else check)(args.length, args.precision)
