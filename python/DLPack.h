// Read-only DLPack views of the buffers of a simulation (D220) and
// writable borrows of them with a commit (D[python-dlpack-write],
// docs/python-dlpack.md), included by Bindings.cpp.
//
// The structures follow the ABI of `dlpack.h` 1.x (dmlc/dlpack,
// https://github.com/dmlc/dlpack), and the exchange the DLPack Python
// specification (https://dmlc.github.io/dlpack/latest/python_spec.html):
// `__dlpack__(*, stream, max_version, dl_device, copy)` returns a capsule
// named "dltensor" (DLManagedTensor) or "dltensor_versioned"
// (DLManagedTensorVersioned); a consumer that takes it renames it
// "used_dltensor[_versioned]" and calls the deleter when its last alias is
// gone; a capsule that no consumer took calls the deleter when it is
// destroyed.
namespace dlpack {
enum DeviceType : int32_t { CPU = 1, CUDA = 2 };
enum TypeCode : uint8_t { Int = 0, UInt = 1, Float = 2 };
struct Device { int32_t type; int32_t id; };
struct DataType { uint8_t code; uint8_t bits; uint16_t lanes; };
struct Tensor {
  void *data;
  Device device;
  int32_t ndim;
  DataType dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byteOffset;
};
struct ManagedTensor {
  Tensor tensor;
  void *context;
  void (*deleter)(ManagedTensor *);
};
struct Version { uint32_t major; uint32_t minor; };
struct ManagedTensorVersioned {
  Version version;
  void *context;
  void (*deleter)(ManagedTensorVersioned *);
  uint64_t flags;
  Tensor tensor;
};
constexpr uint64_t readOnly = 1;
} // namespace dlpack

namespace views {
/// A lease on the buffers of a simulation: while one is alive the
/// simulation refuses what would write or free them. It holds the native
/// simulation, not a Python object, so that it may be released on any
/// thread, with or without the GIL.
struct Lease {
  std::shared_ptr<compiler::Simulation> simulation;
  explicit Lease(std::shared_ptr<compiler::Simulation> s) : simulation(std::move(s)) {}
  Lease(const Lease &) = delete;
  ~Lease() { simulation->releaseLease(); }
};

/// What a View and its Buffers share: the view's own lease, until release().
/// A Borrow shares the same, with the buffers of the host that it owns:
/// the edges of the cell and the values of the tunables, which a consumer
/// writes and a commit takes (D[python-dlpack-write]). A managed tensor
/// holds it as well, so that those buffers outlive the Borrow.
struct ViewState {
  std::shared_ptr<compiler::Simulation> simulation;
  compiler::SimulationView view;
  std::unique_ptr<Lease> lease;
  std::array<double, 3> cell{};
  std::vector<std::vector<double>> tunables;
};

/// One buffer of a view.
struct Buffer {
  std::shared_ptr<ViewState> state;
  void *data = nullptr;
  std::vector<int64_t> shape;
  dlpack::DataType dtype{dlpack::Float, 64, 1};
  dlpack::Device device{dlpack::CPU, 0};
  std::string name;
  /// A buffer of a borrow that a consumer writes, and the field of the
  /// state that its export marks as written
  /// (compiler::Simulation::WrittenPositions, ...), if any.
  bool writable = false;
  unsigned field = 0;
  const char *owner = "View";
};

/// The context of a managed tensor that a consumer was given: its shape,
/// strides, and a lease of its own.
template <class Managed> struct Exported {
  Managed managed{};
  std::vector<int64_t> shape, strides;
  /// Declared before the lease: the lease is released first, and then
  /// whatever of the view only this tensor still holds.
  std::shared_ptr<ViewState> state;
  std::unique_ptr<Lease> lease;
  static void destroy(Managed *self) { delete static_cast<Exported *>(self->context); }
};

inline py::dtype numpyType(const dlpack::DataType &t) {
  if (t.code == dlpack::Int) return py::dtype("int32");
  return py::dtype(t.bits == 32 ? "float32" : "float64");
}

template <class Managed>
static void fill(Exported<Managed> &e, Managed &m, dlpack::Tensor &t, const Buffer &b) {
  e.shape = b.shape;
  e.strides.assign(b.shape.size(), 1);
  for (size_t k = b.shape.size(); k-- > 1;) e.strides[k - 1] = e.strides[k] * b.shape[k];
  t.data = b.data;
  t.device = b.device;
  t.ndim = static_cast<int32_t>(b.shape.size());
  t.dtype = b.dtype;
  t.shape = e.shape.data();
  t.strides = e.strides.data();
  t.byteOffset = 0;
  m.context = &e;
  m.deleter = &Exported<Managed>::destroy;
}

template <class Managed> static void capsuleDestructor(PyObject *capsule, const char *name) {
  // Only a capsule that no consumer took still owns its tensor.
  if (!PyCapsule_IsValid(capsule, name)) return;
  auto *m = static_cast<Managed *>(PyCapsule_GetPointer(capsule, name));
  if (m && m->deleter) m->deleter(m);
}

static py::object exportBuffer(const Buffer &b, py::object stream, py::object maxVersion,
                               py::object dlDevice, py::object copy) {
  if (!b.state->lease)
    throw py::buffer_error(std::string(b.owner) + "." + b.name + (b.owner[0] == 'B'
                                ? ": the borrow has ended; take another with "
                                  "Simulation.borrow()"
                                : ": the view was released; take another "
                                  "with Simulation.view()"));
  if (!copy.is_none() && py::cast<bool>(copy))
    throw py::buffer_error(std::string(b.owner) + "." + b.name + ": a view does not copy (copy=True)");
  if (!dlDevice.is_none()) {
    auto d = py::cast<std::pair<int, int>>(dlDevice);
    if (d.first != b.device.type || d.second != b.device.id)
      throw py::buffer_error(std::string(b.owner) + "." + b.name + ": the buffer is on device (" +
                             std::to_string(b.device.type) + ", " + std::to_string(b.device.id) +
                             "); a view does not copy to another");
  }
  compiler::Simulation &sim = *b.state->simulation;
  if (b.device.type == dlpack::CUDA) {
    int64_t s = stream.is_none() ? 1 : py::cast<int64_t>(stream);
    if (s == 0)
      throw py::value_error(std::string(b.owner) + "." + b.name + ": stream 0 is ambiguous for CUDA; "
                            "pass 1 for the legacy default stream or 2 for the per-thread one");
    if (s < -1)
      throw py::buffer_error(std::string(b.owner) + "." + b.name + ": stream must be -1, 1, 2, or a stream handle");
    if (s == -1) {
      sim.markExported();
    } else {
      py::gil_scoped_release release;
      sim.handOff(static_cast<uint64_t>(s));
    }
  } else if (!stream.is_none() && py::cast<int64_t>(stream) != -1) {
    throw py::buffer_error(std::string(b.owner) + "." + b.name + ": a buffer of the host takes no stream");
  }
  bool versioned = false;
  if (!maxVersion.is_none()) {
    auto v = py::cast<std::pair<int, int>>(maxVersion);
    versioned = v.first >= 1;
  }
  sim.acquireLease();
  auto lease = std::make_unique<Lease>(b.state->simulation);
  // A tensor handed out for writing cannot be told from one written.
  if (b.field) sim.markWritten(b.field);
  if (versioned) {
    auto e = std::make_unique<Exported<dlpack::ManagedTensorVersioned>>();
    e->state = b.state;
    e->lease = std::move(lease);
    auto &m = e->managed;
    m.version = {1, 1};
    m.flags = b.writable ? 0 : dlpack::readOnly;
    fill(*e, m, m.tensor, b);
    PyObject *capsule = PyCapsule_New(&m, "dltensor_versioned", [](PyObject *c) {
      capsuleDestructor<dlpack::ManagedTensorVersioned>(c, "dltensor_versioned");
    });
    if (!capsule) throw py::error_already_set();
    e.release();
    return py::reinterpret_steal<py::object>(capsule);
  }
  auto e = std::make_unique<Exported<dlpack::ManagedTensor>>();
  e->state = b.state;
  e->lease = std::move(lease);
  auto &m = e->managed;
  fill(*e, m, m.tensor, b);
  PyObject *capsule = PyCapsule_New(&m, "dltensor", [](PyObject *c) {
    capsuleDestructor<dlpack::ManagedTensor>(c, "dltensor");
  });
  if (!capsule) throw py::error_already_set();
  e.release();
  return py::reinterpret_steal<py::object>(capsule);
}

/// A view: the buffers of the state at the end of the last part, the
/// particle IDs, and the values of the tunables.
struct View {
  std::shared_ptr<ViewState> state;
  Buffer positions, velocities, forces, ids;
  std::vector<std::pair<std::string, Buffer>> tunables;
};

inline View take(std::shared_ptr<compiler::Simulation> simulation) {
  llvm::Expected<compiler::SimulationView> taken = simulation->takeView();
  if (!taken) raise(taken.takeError());
  // takeView took the lease that this one releases.
  auto lease = std::make_unique<Lease>(simulation);
  compiler::SimulationView v = *taken;
  auto state = std::make_shared<ViewState>();
  state->simulation = simulation;
  state->view = v;
  state->lease = std::move(lease);
  dlpack::Device device{v.onDevice ? dlpack::CUDA : dlpack::CPU, v.onDevice ? v.device : 0};
  auto make = [&](const void *data, std::vector<int64_t> shape, dlpack::DataType dtype,
                  dlpack::Device where, std::string name) {
    Buffer b;
    b.state = state;
    b.data = const_cast<void *>(data);
    b.shape = std::move(shape);
    b.dtype = dtype;
    b.device = where;
    b.name = std::move(name);
    return b;
  };
  int64_t n = static_cast<int64_t>(v.count);
  dlpack::DataType stateType{dlpack::Float, static_cast<uint8_t>(8 * v.stateWidth), 1};
  dlpack::DataType forceType{dlpack::Float, static_cast<uint8_t>(8 * v.forceWidth), 1};
  View view;
  view.state = state;
  view.positions = make(v.positions, {n, 3}, stateType, device, "positions");
  view.velocities = make(v.velocities, {n, 3}, stateType, device, "velocities");
  view.forces = make(v.forces, {n, 3}, forceType, device, "forces");
  view.ids = make(v.ids, {n}, {dlpack::Int, 32, 1}, device, "ids");
  // The values of the tunables, which an update (refused while the lease
  // is alive) replaces.
  const auto &set = simulation->getTunables();
  const auto &values = simulation->getTunableValues();
  for (size_t k = 0; k != set.tunables.size(); ++k)
    view.tunables.emplace_back(set.tunables[k].name,
        make(values[k].data(), {static_cast<int64_t>(values[k].size())},
             {dlpack::Float, 64, 1}, {dlpack::CPU, 0}, "tunables['" + set.tunables[k].name + "']"));
  return view;
}

/// A writable borrow (D[python-dlpack-write]): the buffers of the positions
/// and the velocities where the program keeps them, and buffers of its own
/// with the edges of the cell and the values of the tunables.
struct Borrow {
  std::shared_ptr<ViewState> state;
  Buffer positions, velocities, ids, cell;
  bool hasCell = false;
  std::vector<std::pair<std::string, Buffer>> tunables;
};

inline Borrow borrow(std::shared_ptr<compiler::Simulation> simulation) {
  llvm::Expected<compiler::SimulationView> taken = simulation->takeBorrow();
  if (!taken) raise(taken.takeError());
  auto lease = std::make_unique<Lease>(simulation);
  compiler::SimulationView v = *taken;
  auto state = std::make_shared<ViewState>();
  state->simulation = simulation;
  state->view = v;
  state->lease = std::move(lease);
  state->cell = simulation->getCellEdges();
  state->tunables = simulation->getTunableValues();
  dlpack::Device device{v.onDevice ? dlpack::CUDA : dlpack::CPU, v.onDevice ? v.device : 0};
  auto make = [&](const void *data, std::vector<int64_t> shape, dlpack::DataType dtype,
                  dlpack::Device where, std::string name, bool writable, unsigned field) {
    Buffer b;
    b.state = state;
    b.data = const_cast<void *>(data);
    b.shape = std::move(shape);
    b.dtype = dtype;
    b.device = where;
    b.name = std::move(name);
    b.writable = writable;
    b.field = field;
    b.owner = "Borrow";
    return b;
  };
  int64_t n = static_cast<int64_t>(v.count);
  dlpack::DataType stateType{dlpack::Float, static_cast<uint8_t>(8 * v.stateWidth), 1};
  dlpack::DataType real{dlpack::Float, 64, 1};
  dlpack::Device host{dlpack::CPU, 0};
  Borrow result;
  result.state = state;
  result.positions = make(v.positions, {n, 3}, stateType, device, "positions", true,
                          compiler::Simulation::WrittenPositions);
  result.velocities = make(v.velocities, {n, 3}, stateType, device, "velocities", true,
                           compiler::Simulation::WrittenVelocities);
  result.ids = make(v.ids, {n}, {dlpack::Int, 32, 1}, device, "ids", false, 0);
  result.hasCell = simulation->hasOrthorhombicCell();
  result.cell = make(state->cell.data(), {3}, real, host, "cell", true, 0);
  const auto &set = simulation->getTunables();
  for (size_t k = 0; k != set.tunables.size(); ++k)
    result.tunables.emplace_back(set.tunables[k].name,
        make(state->tunables[k].data(), {static_cast<int64_t>(state->tunables[k].size())},
             real, host, "tunables['" + set.tunables[k].name + "']", true, 0));
  return result;
}

/// The edges of the cell of a borrow, if their bits are not the
/// simulation's, and the tunables whose bits are not.
inline std::optional<std::array<double, 3>> writtenCell(const Borrow &b) {
  if (!b.hasCell) return std::nullopt;
  auto now = b.state->simulation->getCellEdges();
  if (std::memcmp(now.data(), b.state->cell.data(), sizeof now) == 0) return std::nullopt;
  return b.state->cell;
}
inline std::vector<std::pair<std::string, std::vector<double>>> writtenTunables(const Borrow &b) {
  std::vector<std::pair<std::string, std::vector<double>>> changes;
  const auto &now = b.state->simulation->getTunableValues();
  for (size_t k = 0; k != b.tunables.size(); ++k) {
    const auto &mine = b.state->tunables[k];
    if (std::memcmp(now[k].data(), mine.data(), mine.size() * sizeof(double)) != 0)
      changes.emplace_back(b.tunables[k].first, mine);
  }
  return changes;
}

inline void bind(py::module_ &m) {
  py::class_<Buffer>(m, "Buffer")
    .def_property_readonly("shape", [](const Buffer &b) {
      py::tuple t(b.shape.size());
      for (size_t k = 0; k != b.shape.size(); ++k) t[k] = b.shape[k];
      return t;
    })
    .def_property_readonly("dtype", [](const Buffer &b) { return numpyType(b.dtype); })
    .def_property_readonly("device", [](const Buffer &b) {
      return std::make_pair(b.device.type, b.device.id);
    })
    .def_property_readonly("data_ptr", [](const Buffer &b) {
      return reinterpret_cast<uintptr_t>(b.data);
    })
    .def("__dlpack_device__", [](const Buffer &b) {
      return std::make_pair(b.device.type, b.device.id);
    })
    .def("__dlpack__", &exportBuffer, py::kw_only(), py::arg("stream") = py::none(),
         py::arg("max_version") = py::none(), py::arg("dl_device") = py::none(),
         py::arg("copy") = py::none())
    .def("__repr__", [](const Buffer &b) {
      std::string shape;
      for (size_t k = 0; k != b.shape.size(); ++k)
        shape += (k ? ", " : "") + std::to_string(b.shape[k]);
      if (b.shape.size() == 1) shape += ",";
      return "Buffer(" + b.name + ", shape=(" + shape + "), dtype=" +
             py::cast<std::string>(py::str(numpyType(b.dtype))) + ", device=(" +
             std::to_string(b.device.type) + ", " + std::to_string(b.device.id) + "))";
    });
  py::class_<View>(m, "View")
    .def_readonly("positions", &View::positions)
    .def_readonly("velocities", &View::velocities)
    .def_readonly("forces", &View::forces)
    .def_readonly("ids", &View::ids)
    .def_property_readonly("tunables", [](const View &v) {
      py::dict d;
      for (const auto &entry : v.tunables) d[py::str(entry.first)] = py::cast(entry.second);
      return d;
    })
    .def_property_readonly("step", [](const View &v) { return v.state->view.step; })
    .def_property_readonly("time", [](const View &v) { return v.state->view.time; })
    .def_property_readonly("velocity_offset", [](const View &v) { return v.state->view.velocityOffset; })
    .def_property_readonly("device", [](const View &v) {
      return std::make_pair(v.positions.device.type, v.positions.device.id);
    })
    .def_property_readonly("valid", [](const View &v) {
      return v.state->simulation->getGeneration() == v.state->view.generation;
    })
    .def_property_readonly("released", [](const View &v) { return !v.state->lease; })
    .def("release", [](View &v) { v.state->lease.reset(); })
    .def("__enter__", [](py::object self) { return self; })
    .def("__exit__", [](View &v, py::args) { v.state->lease.reset(); return false; })
    .def("__repr__", [](const View &v) {
      return "View(step=" + std::to_string(v.state->view.step) + ", particles=" +
             std::to_string(v.state->view.count) + (v.state->lease ? "" : ", released") + ")";
    });
  static auto live = [](const Borrow &b) { return static_cast<bool>(b.state->lease); };
  py::class_<Borrow>(m, "Borrow")
    .def_readonly("positions", &Borrow::positions)
    .def_readonly("velocities", &Borrow::velocities)
    .def_readonly("ids", &Borrow::ids)
    .def_property_readonly("cell", [](const Borrow &b) {
      if (!b.hasCell)
        throw UnsupportedError("Borrow.cell: a borrow takes the edges of an orthorhombic "
                               "periodic cell; the tilts of a triclinic cell are not "
                               "supported yet (#206)");
      return b.cell;
    })
    .def_property_readonly("tunables", [](const Borrow &b) {
      py::dict d;
      for (const auto &entry : b.tunables) d[py::str(entry.first)] = py::cast(entry.second);
      return d;
    })
    .def_property_readonly("step", [](const Borrow &b) { return b.state->view.step; })
    .def_property_readonly("time", [](const Borrow &b) { return b.state->view.time; })
    .def_property_readonly("velocity_offset", [](const Borrow &b) { return b.state->view.velocityOffset; })
    .def_property_readonly("device", [](const Borrow &b) {
      return std::make_pair(b.positions.device.type, b.positions.device.id);
    })
    .def_property_readonly("live", [](const Borrow &b) { return live(b); })
    .def_property_readonly("written", [](const Borrow &b) {
      py::list names;
      if (!live(b)) return py::tuple(names);
      unsigned written = b.state->simulation->getWritten();
      if (written & compiler::Simulation::WrittenPositions) names.append("positions");
      if (written & compiler::Simulation::WrittenVelocities) names.append("velocities");
      if (writtenCell(b)) names.append("cell");
      if (!writtenTunables(b).empty()) names.append("tunables");
      return py::tuple(names);
    })
    .def("commit", [](Borrow &b) {
      if (!live(b))
        throw SimulationError("Borrow.commit: the borrow has ended");
      compiler::Simulation &sim = *b.state->simulation;
      auto cell = writtenCell(b);
      auto changes = writtenTunables(b);
      std::optional<llvm::Expected<std::vector<std::string>>> fields;
      {
        py::gil_scoped_release release;
        fields.emplace(sim.commitBorrow(cell, changes));
      }
      // Committed, or undone after an evaluation that failed: the borrow
      // has ended either way. A refusal leaves it live.
      if (!sim.isBorrowed()) b.state->lease.reset();
      if (!*fields) raise(fields->takeError());
      py::list names;
      for (const std::string &name : **fields) names.append(name);
      return py::tuple(names);
    })
    .def("abandon", [](Borrow &b) { b.state->lease.reset(); })
    .def("__enter__", [](py::object self) { return self; })
    .def("__exit__", [](Borrow &b, py::args) { b.state->lease.reset(); return false; })
    .def("__repr__", [](const Borrow &b) {
      return "Borrow(step=" + std::to_string(b.state->view.step) + ", particles=" +
             std::to_string(b.state->view.count) + (live(b) ? "" : ", ended") + ")";
    });
}
} // namespace views
