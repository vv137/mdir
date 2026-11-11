"""Compares two HDF5 files by content: every group, every dataset (its
type, shape, and bytes), and every attribute, since the library writes
different bytes for the same content from run to run. Prints each
difference and the verdict.

    compare_h5_contents.py A B [NAME@ATTRIBUTE...]

The names after the files are attributes left out of the comparison."""
import sys

import h5py
import numpy

a, b = (h5py.File(path, 'r') for path in sys.argv[1:3])
ignored = set(sys.argv[3:])
differences = []
datasets = 0


def same(x, y):
    x, y = numpy.asarray(x), numpy.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def walk(x, y, name):
    global datasets
    if set(x.attrs) != set(y.attrs):
        differences.append(f'{name}: attributes {sorted(set(x.attrs) ^ set(y.attrs))}')
    for key in set(x.attrs) & set(y.attrs):
        if f'{name}@{key}' not in ignored and not same(x.attrs[key], y.attrs[key]):
            differences.append(f'{name}@{key}: {x.attrs[key]!r} against {y.attrs[key]!r}')
    if isinstance(x, h5py.Dataset) or isinstance(y, h5py.Dataset):
        datasets += 1
        if not (isinstance(x, h5py.Dataset) and isinstance(y, h5py.Dataset)) or not same(x[()], y[()]):
            differences.append(f'{name}: the values differ')
        return
    if set(x) != set(y):
        differences.append(f'{name}: members {sorted(set(x) ^ set(y))}')
    for key in sorted(set(x) & set(y)):
        walk(x[key], y[key], f'{name}/{key}')


walk(a, b, '')
for difference in differences:
    print(difference)
print(f'{datasets} datasets: ' + ('identical contents' if not differences
                                 else f'{len(differences)} differences'))
sys.exit(1 if differences else 0)
