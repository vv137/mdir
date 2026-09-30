#!/usr/bin/env python3
"""Fits the polynomial of the approximation of erfc that md-exec-approximate
writes into kernels in f32 under fast_math:

    erfc(x) ~ exp(-x^2) P(t),  t = 1 / (1 + x / 2),  0 <= x <= 6,

P of degree 9, fitted by weighted least squares (weights 1 / erfcx(x), so
that the error is relative) on Chebyshev nodes in t, where
erfcx(x) = exp(x^2) erfc(x). Prints the coefficients of P from the constant
term up, for lib/Dialect/MDExec/Transforms/Approximate.cpp, and the largest
relative error of erfc on [0, 6].

    python3 scripts/fit-erfc.py    (needs NumPy)"""

import math

import numpy as np
from numpy.polynomial import chebyshev

DEGREE = 9
RANGE = 6.0
SLOPE = 0.5


def erfcx(x):
    return np.array([math.erfc(v) * math.exp(v * v) for v in x])


def main():
    t_end = 1.0 / (1.0 + SLOPE * RANGE)
    nodes = np.cos(np.pi * (np.arange(4000) + 0.5) / 4000)
    t = t_end + (nodes + 1.0) * (1.0 - t_end) / 2.0
    x = (1.0 / t - 1.0) / SLOPE
    target = erfcx(x)
    fit = chebyshev.Chebyshev.fit(t, target, DEGREE, domain=[t_end, 1.0],
                                  w=1.0 / target)
    coefficients = fit.convert(kind=np.polynomial.Polynomial,
                               domain=[-1.0, 1.0]).coef
    for c in coefficients:
        print(f"{c:.17g},")

    xs = np.linspace(0.0, RANGE, 60001)
    exact = np.array([math.erfc(v) for v in xs])
    ts = 1.0 / (1.0 + SLOPE * xs)
    p64 = np.polynomial.Polynomial(coefficients)(ts)
    xs32 = xs.astype(np.float32)
    ts32 = np.float32(1.0) / (np.float32(1.0) + np.float32(SLOPE) * xs32)
    p32 = np.zeros_like(ts32)
    for c in coefficients.astype(np.float32)[::-1]:
        p32 = (p32 * ts32 + c).astype(np.float32)
    # With P in f32 and the exponential exact, the error of the
    # approximation; with the argument of the exponential rounded to f32 as
    # well, what a kernel gets (the rounding of x^2, about x^2 2^-24, is in
    # the exponential of the derivative too).
    only = p32.astype(np.float64) * np.exp(-xs * xs)
    rounded = p32.astype(np.float64) * np.exp(
        (-(xs32 * xs32)).astype(np.float64))
    print("# largest relative error of erfc on [0, 6]: "
          f"{np.max(np.abs(p64 * np.exp(-xs * xs) - exact) / exact):.2e} "
          f"(P in f64), {np.max(np.abs(only - exact) / exact):.2e} (P in "
          f"f32), {np.max(np.abs(rounded - exact) / exact):.2e} (and the "
          "argument of exp in f32)")


if __name__ == "__main__":
    main()
