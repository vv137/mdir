"""Writes the control file of mixture.toml with its Lennard-Jones as a
tabulated function of r/sigma, x^-12 - x^-6 at 381 points from 0.8 to 2.7,
which the expression of the pair term calls (D138).

Usage: tabulated_mixture.py MIXTURE_TOML > CONTROL"""
import sys

lo, hi, n = 0.8, 2.7, 381
values = []
for i in range(n):
    x = lo + (hi - lo) * i / (n - 1)
    values.append("%.17g" % (x ** -12 - x ** -6))
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    if line.startswith("expression"):
        line = 'expression = "4*epsilon*lj(r/sigma)"\n'
    if line.startswith('coordinates'):
        line = line.replace('"Inputs/', '"')
    sys.stdout.write(line)
print("\n[[energy.function]]\nname = \"lj\"\nmin = %r\nmax = %r\nvalues = [%s]" % (lo, hi, ", ".join(values)))
