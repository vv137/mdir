"""MDIR's Python interface: models, compilation, and simulations.

The interface is the native extension `mdir._core` (D192); this package
gives its names under `mdir` and carries, in an installed wheel, the
runtime libraries and libdevice that the extension finds beside itself
(D228, docs/python-package.md).
"""
from . import _core
from ._core import *  # noqa: F401,F403

__version__ = _core.__version__


def _publish():
    """Name the extension's classes, enumerations, and exceptions `mdir.X`.

    pybind11 names them after the module that defines them, `mdir._core`;
    tracebacks and reprs say `mdir.InputError`, the name a user writes.
    """
    for name in dir(_core):
        value = getattr(_core, name)
        if isinstance(value, type) and value.__module__ == _core.__name__:
            try:
                value.__module__ = __name__
            except (AttributeError, TypeError):
                pass


_publish()
del _publish
