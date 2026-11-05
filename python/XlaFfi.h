// The handlers of the XLA FFI for the JAX adapter of the frame evaluator
// (D371, docs/python-frames.md, `mdir.jax`).
#ifndef MDIR_PYTHON_XLAFFI_H
#define MDIR_PYTHON_XLAFFI_H

#include "mdir/Compiler/Simulation.h"
#include <pybind11/pybind11.h>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace mdir::python::xla {

/// Gives the simulation of a frame evaluator a key of the process, under
/// which the handlers find it, and returns the key with the fingerprint
/// of its program, the two static attributes of a call. The registry holds
/// the simulation until `release`; an executable that was instantiated
/// with it holds it as well.
std::pair<int64_t, std::string>
registerProgram(std::shared_ptr<compiler::Simulation> simulation,
                const model::PreparedModel &prepared);

/// Adds `_xla_ffi_targets`, `_xla_ffi_types`, `_xla_ffi_api_version`, and
/// `_xla_ffi_release` to the module.
void addBindings(pybind11::module_ &m);

} // namespace mdir::python::xla

#endif
