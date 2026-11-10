"""MDIR's Python interface: models, compilation, and simulations.

The interface is the native extension `mdir._core` (D192); this package
gives its names under `mdir` and carries, in an installed wheel, the
runtime libraries and libdevice that the extension finds beside itself
(D228, docs/python-package.md).
"""
from . import _core, _physical_constants
from ._core import *  # noqa: F401,F403

__version__ = _core.__version__

#: Boltzmann's constant in kJ/(mol K), the molar gas constant k N_A of the
#: compiled programs (docs/conventions.md, Section 1), for the weights of a
#: reweighting (docs/python-frames.md).
KB = _physical_constants.MOLAR_GAS_CONSTANT


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

from ._frames import FrameEnergies, FrameEvaluator, FrameGradient, FrameTangent  # noqa: E402,F401
