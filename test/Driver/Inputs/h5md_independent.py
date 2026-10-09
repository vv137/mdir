"""The H5MD files of MDIR (D[h5md-reporter]) read by programs that are not
MDIR: h5py with the paths of the specification written out by hand, and
the H5MD reader of MDAnalysis if it is installed, which opens the files
with `strings = "variable"` as they are written. The states to compare
with are in `reference.npz`, which `python_h5md.py reference` writes beside
the files. This script does not import mdir."""
import pathlib
import sys

import h5py
import numpy as np

work = pathlib.Path(sys.argv[1])
reference = np.load(work / "reference.npz")


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def same_object(file, a, b):
    """Whether two paths are hard links to one object."""
    return h5py.h5o.get_info(file[a].id).addr == h5py.h5o.get_info(file[b].id).addr


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def check(name, kind, triclinic=False, velocities=False, forces=False, tunables=False,
          variable=False):
    with h5py.File(work / f"{name}.h5md", "r") as f:
        # The metadata of H5MD 1.1 and the units module 1.0.
        h5md = f["h5md"]
        assert list(h5md.attrs["version"]) == [1, 1]
        assert text(h5md["author"].attrs["name"]) == "unknown"
        assert text(h5md["creator"].attrs["name"]) == "MDIR"
        assert text(h5md["creator"].attrs["version"])
        units = h5md["modules/units"]
        assert list(units.attrs["version"]) == [1, 0] and text(units.attrs["system"]) == "SI"
        # Every string but the `unit` attributes is of fixed length in both
        # forms; the `unit` attributes are of the form asked for.
        for attribute in (h5md["author"].attrs.get_id("name"), h5md["creator"].attrs.get_id("name"),
                          units.attrs.get_id("system"), f["particles/all/box"].attrs.get_id("boundary"),
                          f["parameters/mdir"].attrs.get_id("strings")):
            assert not attribute.get_type().is_variable_str()
        assert text(f["parameters/mdir"].attrs["strings"]) == ("variable" if variable else "fixed")
        labeled = []
        f.visititems(lambda _, item: labeled.append(item) if "unit" in item.attrs else None)
        assert len(labeled) >= 5
        for item in labeled:
            assert item.attrs.get_id("unit").get_type().is_variable_str() == variable, item.name
            assert isinstance(item.attrs["unit"], str if variable else bytes), item.name

        group = f["particles/all"]
        box = group["box"]
        assert box.attrs["dimension"] == 3
        assert [text(b) for b in box.attrs["boundary"]] == ["periodic"] * 3
        position = group["position"]
        steps, times, values = position["step"], position["time"], position["value"]
        count = len(reference[f"{name}/steps"])
        particles = reference[f"{name}/positions"].shape[1]
        assert steps.dtype == np.int64 and steps.shape == (count,)
        assert times.dtype == np.float64 and times.shape == (count,)
        assert values.dtype == kind and values.shape == (count, particles, 3)
        assert values.maxshape == (None, particles, 3) and values.compression is None
        assert text(values.attrs["unit"]) == "nm" and text(times.attrs["unit"]) == "ps"
        assert bits(steps[:], reference[f"{name}/steps"])
        assert bits(times[:], reference[f"{name}/times"])
        assert np.all(np.diff(steps[:]) > 0)
        assert bits(values[:], reference[f"{name}/positions"].astype(kind))

        edges = box["edges"]
        assert same_object(f, "particles/all/box/edges/step", "particles/all/position/step")
        assert same_object(f, "particles/all/box/edges/time", "particles/all/position/time")
        assert text(edges["value"].attrs["unit"]) == "nm" and edges["value"].dtype == np.float64
        assert edges["value"].shape == ((count, 3, 3) if triclinic else (count, 3))
        cells = reference[f"{name}/cells"]
        if triclinic:
            assert bits(edges["value"][:], cells)
            assert np.all(edges["value"][:, 0, 1:] == 0) and np.all(edges["value"][:, 1, 2] == 0)
        else:
            assert bits(edges["value"][:], np.ascontiguousarray(np.diagonal(cells, axis1=1, axis2=2)))

        for element, asked, unit in (("velocity", velocities, "nm ps-1"),
                                     ("force", forces, "kJ mol-1 nm-1")):
            assert (element in group) == asked
            if not asked:
                continue
            value = group[element]["value"]
            assert value.dtype == kind and text(value.attrs["unit"]) == unit
            assert same_object(f, f"particles/all/{element}/step", "particles/all/position/step")
            assert same_object(f, f"particles/all/{element}/time", "particles/all/position/time")
            assert bits(value[:], reference[f"{name}/{element}"].astype(kind))
        assert "id" not in group
        assert group["mass"].shape == (particles,) and text(group["mass"].attrs["unit"]) == "g mol-1"
        assert bits(group["mass"][:], reference[f"{name}/masses"])
        assert group["species"].shape == (particles,) and group["species"].dtype == np.int32

        energy = f["observables/potential_energy"]
        assert same_object(f, "observables/potential_energy/step", "particles/all/position/step")
        assert text(energy["value"].attrs["unit"]) == "kJ mol-1"
        assert bits(energy["value"][:], reference[f"{name}/potential"])
        assert ("tunables_version" in f["observables"]) == tunables
        parameters = f["parameters/mdir"]
        assert parameters.attrs["trajectory_format"] == 1 and parameters.attrs["period"] == 10
        assert parameters.attrs["timestep"] == 0.0005
        assert text(parameters.attrs["front_end"]) == "python"
        # The earliest file format: no mark of a writer to clear after a kill.
        assert f.id.get_create_plist().get_version()[0] == 0
    return count, particles


for name, kind, options in (
        ("ref-nve", np.float64, {}),
        ("ref-nve-f32", np.float32, {}),
        ("ref-nve-full", np.float64, {"velocities": True, "forces": True}),
        ("ref-npt", np.float64, {}),
        ("ref-tri", np.float64, {"triclinic": True, "velocities": True}),
        ("var-nve-full", np.float64, {"velocities": True, "forces": True, "variable": True}),
        ("var-npt", np.float64, {"variable": True}),
        ("var-tri", np.float64, {"triclinic": True, "velocities": True, "variable": True})):
    count, particles = check(name, kind, **options)
    print(f"h5py {name}: {count} frames of {particles} particles; the paths, types, units, and "
          f"hard links of H5MD 1.1, and the states to the bit")
# The two forms hold the same numbers.
for fixed, variable in (("ref-nve-full", "var-nve-full"), ("ref-npt", "var-npt"), ("ref-tri", "var-tri")):
    with h5py.File(work / f"{fixed}.h5md", "r") as a, h5py.File(work / f"{variable}.h5md", "r") as b:
        names = []
        a.visititems(lambda name, item: names.append(name) if isinstance(item, h5py.Dataset) else None)
        for name in names:
            assert bits(a[name][()], b[name][()]), name
print("h5py: the files with fixed-length and with variable-length unit strings hold the same datasets")

# A killed run: the file opens in another library build and holds whole frames.
with h5py.File(work / "killed.h5md", "r") as f:
    lengths = {f[p].shape[0] for p in ("particles/all/position/value", "particles/all/position/step",
                                       "particles/all/position/time", "particles/all/box/edges/value",
                                       "observables/potential_energy/value")}
    assert lengths == {3}, lengths
    assert bits(f["particles/all/position/value"][:], reference["ref-nve/positions"][:3])
print("h5py killed: 3 frames, every dataset of the same length")

try:
    import MDAnalysis as mda
    from MDAnalysis.lib.mdamath import triclinic_box
except ImportError:
    print("MDAnalysis is not installed: the files with variable strings were not read by it")
    print("MDAnalysis is not installed: the files with fixed strings were not read by it")
    sys.exit(0)

versions = f"MDAnalysis {mda.__version__} with h5py {h5py.__version__}"
for name, triclinic, velocities, forces in (("nve-full", False, True, True),
                                            ("npt", False, False, False),
                                            ("tri", True, True, False)):
    positions, cells = reference[f"var-{name}/positions"], reference[f"var-{name}/cells"]

    def universe():
        return mda.Universe.empty(positions.shape[1], trajectory=True, velocities=velocities,
                                  forces=forces)
    # The file with `strings = "variable"`, as written: without the unit
    # conversion of the reader (nm, in the f32 of its timesteps) and with
    # it (Å, Å/ps, kJ/(mol Å)).
    stored, converted = universe(), universe()
    stored.load_new(str(work / f"var-{name}.h5md"), format="H5MD", convert_units=False)
    converted.load_new(str(work / f"var-{name}.h5md"), format="H5MD")
    assert len(positions) == len(stored.trajectory) == len(converted.trajectory)
    for k, (raw, ts) in enumerate(zip(stored.trajectory, converted.trajectory)):
        assert raw.data["step"] == reference[f"var-{name}/steps"][k]
        assert abs(raw.time - reference[f"var-{name}/times"][k]) < 1e-12
        assert bits(raw.positions, positions[k].astype(np.float32))
        assert np.allclose(ts.positions, positions[k] * 10.0, rtol=1e-6, atol=0)
        expected = triclinic_box(*cells[k].astype(np.float32))
        assert np.allclose(raw.dimensions, expected, rtol=1e-6), (raw.dimensions, expected)
        assert np.allclose(ts.dimensions[:3], expected[:3] * 10.0, rtol=1e-6)
        assert (raw.dimensions[3:] != 90).any() == triclinic
        if velocities:
            assert bits(raw.velocities, reference[f"var-{name}/velocity"][k].astype(np.float32))
            assert np.allclose(ts.velocities, reference[f"var-{name}/velocity"][k] * 10.0, rtol=1e-6)
        if forces:
            assert bits(raw.forces, reference[f"var-{name}/force"][k].astype(np.float32))
            assert np.allclose(ts.forces, reference[f"var-{name}/force"][k] / 10.0, rtol=1e-6)
        assert raw.data["potential_energy"] == reference[f"var-{name}/potential"][k]
    print(f"{versions}, variable strings, {name}: {len(positions)} frames as written: positions"
          f"{', velocities' if velocities else ''}{', forces' if forces else ''}, the cell, the "
          f"step, the time, and the potential energy, with and without its unit conversion")
    # The file with `strings = "fixed"`: what this version does is recorded,
    # and either outcome passes.
    try:
        fixed = universe()
        fixed.load_new(str(work / f"ref-{name}.h5md"), format="H5MD", convert_units=False)
        assert len(fixed.trajectory) == len(positions)
        for k, raw in enumerate(fixed.trajectory):
            assert bits(raw.positions, positions[k].astype(np.float32))
        print(f"{versions}, fixed strings, {name}: reads the {len(positions)} frames")
    except RuntimeError as error:
        print(f"{versions}, fixed strings, {name}: stops with the message: {str(error).splitlines()[0]}")
