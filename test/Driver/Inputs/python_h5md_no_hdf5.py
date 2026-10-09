"""Without HDF5 (D239): UnsupportedError from the reporter and
the reader of a trajectory in H5MD."""
import sys

import mdir

work = sys.argv[1]
for call in (lambda: mdir.H5MDReporter(work + "/x.h5md", 10),
             lambda: mdir.read_h5md(work + "/x.h5md")):
    try:
        call()
    except mdir.UnsupportedError as error:
        assert "has no HDF5, which a trajectory in H5MD needs" in str(error), error
    else:
        raise AssertionError("expected UnsupportedError")
print("a trajectory in H5MD without HDF5 raises UnsupportedError")
