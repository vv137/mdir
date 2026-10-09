"""The H5MD files of MDIR (D[h5md-reporter]) read by programs that are not
MDIR: h5py with the paths of the specification written out by hand, and
the H5MD reader of MDAnalysis if it is installed. The states to compare
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


def check(name, kind, triclinic=False, velocities=False, forces=False, tunables=False):
    with h5py.File(work / f"{name}.h5md", "r") as f:
        # The metadata of H5MD 1.1 and the units module 1.0.
        h5md = f["h5md"]
        assert list(h5md.attrs["version"]) == [1, 1]
        assert text(h5md["author"].attrs["name"]) == "unknown"
        assert text(h5md["creator"].attrs["name"]) == "MDIR"
        assert text(h5md["creator"].attrs["version"])
        units = h5md["modules/units"]
        assert list(units.attrs["version"]) == [1, 0] and text(units.attrs["system"]) == "SI"
        for attribute in (h5md["author"].attrs.get_id("name"), h5md["creator"].attrs.get_id("name")):
            assert not attribute.get_type().is_variable_str()

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
        ("ref-tri", np.float64, {"triclinic": True, "velocities": True})):
    count, particles = check(name, kind, **options)
    print(f"h5py {name}: {count} frames of {particles} particles; the paths, types, units, and "
          f"hard links of H5MD 1.1, and the states to the bit")

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
    print("MDAnalysis is not installed")
    sys.exit(0)

import shutil

for name, triclinic, velocities, forces in (("ref-nve-full", False, True, True),
                                            ("ref-npt", False, False, False),
                                            ("ref-tri", True, True, False)):
    positions, cells = reference[f"{name}/positions"], reference[f"{name}/cells"]

    def universe():
        return mda.Universe.empty(positions.shape[1], trajectory=True, velocities=velocities,
                                  forces=forces)
    # The reader looks the unit strings up as Python strings, and h5py gives
    # the fixed-length strings that H5MD asks for as bytes: it refuses the
    # file as MDIR writes it, with and without its unit conversion.
    try:
        universe().load_new(str(work / f"{name}.h5md"), format="H5MD", convert_units=False)
        as_written = "reads the file as written"
    except RuntimeError as error:
        assert "is not recognized by H5MDReader" in str(error), error
        as_written = "refuses the fixed-length unit strings of the file as written"
    # The same file with its unit attributes written again as
    # variable-length strings, and nothing else changed.
    copy = work / f"{name}-vlen.h5md"
    shutil.copyfile(work / f"{name}.h5md", copy)
    with h5py.File(copy, "r+") as f:
        def rewrite(_, item):
            if "unit" in item.attrs:
                unit = text(item.attrs["unit"])
                del item.attrs["unit"]
                item.attrs["unit"] = unit
        f.visititems(rewrite)
    stored, converted = universe(), universe()
    stored.load_new(str(copy), format="H5MD", convert_units=False)
    converted.load_new(str(copy), format="H5MD")
    assert len(positions) == len(stored.trajectory) == len(converted.trajectory)
    for k, (raw, ts) in enumerate(zip(stored.trajectory, converted.trajectory)):
        assert raw.data["step"] == reference[f"{name}/steps"][k]
        assert abs(raw.time - reference[f"{name}/times"][k]) < 1e-12
        # The numbers of the file in the f32 of its timesteps (nm), and in
        # its own units (Å, Å/ps, kJ/(mol Å)).
        assert bits(raw.positions, positions[k].astype(np.float32))
        assert np.allclose(ts.positions, positions[k] * 10.0, rtol=1e-6, atol=0)
        expected = triclinic_box(*cells[k].astype(np.float32))
        assert np.allclose(raw.dimensions, expected, rtol=1e-6), (raw.dimensions, expected)
        assert np.allclose(ts.dimensions[:3], expected[:3] * 10.0, rtol=1e-6)
        assert (raw.dimensions[3:] != 90).any() == triclinic
        if velocities:
            assert bits(raw.velocities, reference[f"{name}/velocity"][k].astype(np.float32))
            assert np.allclose(ts.velocities, reference[f"{name}/velocity"][k] * 10.0, rtol=1e-6)
        if forces:
            assert bits(raw.forces, reference[f"{name}/force"][k].astype(np.float32))
            assert np.allclose(ts.forces, reference[f"{name}/force"][k] / 10.0, rtol=1e-6)
        assert raw.data["potential_energy"] == reference[f"{name}/potential"][k]
    print(f"MDAnalysis {mda.__version__} {name}: {as_written}; with the unit attributes as "
          f"variable-length strings, {len(positions)} frames: positions"
          f"{', velocities' if velocities else ''}{', forces' if forces else ''}, the cell, the "
          f"step, the time, and the potential energy, with and without its unit conversion")
